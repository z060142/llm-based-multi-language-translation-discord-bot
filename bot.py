# -*- coding: utf-8 -*-
import discord
import os
import yaml
import logging
import asyncio
import json
import openai
from openai import AsyncOpenAI
import re
from collections import defaultdict
import aiosqlite

# --- Configuration ---
CONFIG_FILE = 'config.yaml'
DB_FILE = 'translator_data.db'
# Configure logging (Set back to INFO or keep DEBUG if preferred)
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('TranslationBot')

# OpenAI compatible base URL for OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Core Bot Class ---
class TranslationBot(discord.Client):
    """
    Translation bot class inheriting from discord.Client.
    Includes message edit tracking (using SQLite via on_raw_message_edit)
    and channel type support.
    """
    def __init__(self, *, intents: discord.Intents, config_path: str):
        # Ensure necessary intents are enabled, especially messages for content access
        if not intents.messages:
             logger.warning("discord.Intents.messages is not enabled. Bot might miss message events.")
        if not intents.message_content:
             logger.warning("discord.Intents.message_content is not enabled. Bot cannot read message content.")

        super().__init__(intents=intents)
        self.config_path = config_path
        self.channel_language_map = {}
        self.channel_type_map = {}
        self.language_channel_map = defaultdict(list)
        self.api_keys = {}
        self.settings = {
            'translation_model': 'google/gemma-3-27b-it',
            'max_retries': 2,
            'retry_delay': 3
        }
        self.openai_client: AsyncOpenAI | None = None
        self.db: aiosqlite.Connection | None = None

    async def setup_hook(self) -> None:
        """Initialize database and load config."""
        await self.initialize_database()
        self.load_config()
        readable_channels = [cid for cid, type in self.channel_type_map.items() if type in ['standard', 'read_only']]
        logger.info(f'Configuration loaded. Listening for messages in {len(readable_channels)} channels.')
        logger.info(f"Using model '{self.settings.get('translation_model')}' for translation.")
        openrouter_key = self.api_keys.get('openrouter_key')
        if openrouter_key and openrouter_key != 'YOUR_OPENROUTER_API_KEY':
            self.openai_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_key,
            )
            logger.info("OpenAI client initialized for OpenRouter.")
        else:
            logger.error("Failed to initialize OpenAI client: Missing or invalid OpenRouter API Key.")

    async def close(self) -> None:
        """Close database connection and shut down."""
        if self.db:
            try:
                await self.db.commit()
                logger.info("Final database commit before closing.")
            except Exception as e_commit:
                 logger.error(f"Error during final commit: {e_commit}")
            await self.db.close()
            logger.info("Database connection closed.")
        await super().close()
        logger.info("Bot is shutting down.")

    async def initialize_database(self):
        """Connects to the SQLite database and creates tables if they don't exist."""
        try:
            self.db = await aiosqlite.connect(DB_FILE)
            await self.db.execute("PRAGMA journal_mode=WAL;")
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS message_map (
                    original_message_id INTEGER NOT NULL,
                    target_language TEXT NOT NULL,
                    translated_message_id INTEGER NOT NULL,
                    target_channel_id INTEGER NOT NULL,
                    PRIMARY KEY (original_message_id, target_language)
                )
            """)
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_original_message_id ON message_map (original_message_id)
            """)
            await self.db.commit()
            logger.info(f"Successfully connected to database '{DB_FILE}' and ensured table exists.")
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            self.db = None

    def load_config(self):
        """Load settings from the YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if not config_data:
                    logger.error(f"Configuration file '{self.config_path}' is empty.")
                    self.create_example_config()
                    logger.critical("Example configuration file created. Please fill in your information and restart the Bot.")
                    return

                # Load channel settings
                self.channel_language_map = {}
                self.channel_type_map = {}
                self.language_channel_map = defaultdict(list)
                total_channels_loaded = 0

                if 'channels' in config_data and isinstance(config_data['channels'], dict):
                    channel_configs = config_data['channels']
                    for channel_type, languages in channel_configs.items():
                        if channel_type not in ['standard', 'read_only', 'write_only']:
                            logger.warning(f"Unknown channel type '{channel_type}' in config. Skipping.")
                            continue
                        if not isinstance(languages, dict):
                             logger.warning(f"Expected dictionary for languages under channel type '{channel_type}', got {type(languages)}. Skipping.")
                             continue

                        logger.info(f"Loading '{channel_type}' channels...")
                        for lang_code, channel_ids in languages.items():
                            if not isinstance(channel_ids, list):
                                channel_ids = [channel_ids]

                            for channel_id in channel_ids:
                                try:
                                    ch_id_int = int(channel_id)
                                    if ch_id_int in self.channel_language_map:
                                        logger.warning(f"Channel ID {ch_id_int} was already configured as language '{self.channel_language_map[ch_id_int]}' (type: {self.channel_type_map.get(ch_id_int, 'unknown')}). Overwriting with language '{lang_code}' (type: {channel_type}).")
                                    self.channel_language_map[ch_id_int] = lang_code
                                    self.channel_type_map[ch_id_int] = channel_type
                                    self.language_channel_map[lang_code].append(ch_id_int)
                                    total_channels_loaded += 1
                                    # Keep this debug log if useful
                                    # logger.debug(f"  Loaded Channel ID: {ch_id_int}, Lang: {lang_code}, Type: {channel_type}")
                                except ValueError:
                                    logger.warning(f"Invalid channel ID '{channel_id}' found under language '{lang_code}' (type: {channel_type}). Skipping.")
                    logger.info(f"Successfully loaded {total_channels_loaded} channel configurations across all types.")
                    # Keep these debug logs if useful
                    # logger.debug(f"Final channel_language_map: {self.channel_language_map}")
                    # logger.debug(f"Final channel_type_map: {self.channel_type_map}")
                    # logger.debug(f"Final language_channel_map: {dict(self.language_channel_map)}")
                else:
                    logger.warning(f"Missing or invalid 'channels' section in configuration file '{self.config_path}'. No channels loaded.")

                # Load API keys
                if 'api_keys' in config_data and isinstance(config_data['api_keys'], dict):
                    self.api_keys = config_data['api_keys']
                    logger.info("Successfully loaded API key settings.")
                else: logger.warning(f"Missing or invalid 'api_keys' section in configuration file '{self.config_path}'.")

                # Load other settings
                if 'settings' in config_data and isinstance(config_data['settings'], dict):
                    self.settings['translation_model'] = config_data['settings'].get('translation_model', self.settings['translation_model'])
                    self.settings['max_retries'] = config_data['settings'].get('max_retries', self.settings['max_retries'])
                    self.settings['retry_delay'] = config_data['settings'].get('retry_delay', self.settings['retry_delay'])
                    logger.info("Successfully loaded other settings.")
                else: logger.info("No 'settings' section found, using default settings.")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}.")
            self.create_example_config()
            logger.critical("Example configuration file created. Please fill in your Discord Token, OpenRouter Key, and channel IDs, then restart the Bot.")
        except yaml.YAMLError as e: logger.error(f"Error parsing configuration file '{self.config_path}': {e}")
        except Exception as e: logger.error(f"Unexpected error loading configuration: {e}", exc_info=True)


    def create_example_config(self):
        """Create an example configuration file."""
        default_config = {
            'api_keys': {
                'discord_token': 'YOUR_DISCORD_BOT_TOKEN',
                'openrouter_key': 'YOUR_OPENROUTER_API_KEY'
            },
            'channels': {
                'standard': {
                    'en': [123456789012345678],
                    'zh-TW': [987654321098765432]
                },
                'read_only': {
                    'ja': [111111111111111111]
                },
                'write_only': {
                    'ko': [222222222222222222]
                }
            },
            'settings': {
                'translation_model': 'google/gemma-3-27b-it',
                'max_retries': 2,
                'retry_delay': 3
            }
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f: yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            logger.info(f"Example configuration file '{self.config_path}' created with new structure. Please fill in your information.")
        except Exception as e: logger.error(f"Error creating example configuration file '{self.config_path}': {e}")


    async def on_ready(self):
        """Called when the Bot is ready."""
        logger.info(f'Logged in successfully as {self.user} (ID: {self.user.id})')
        logger.info('Bot is ready.')

    async def on_message(self, message: discord.Message):
        """Handle new message events, considering channel types."""
        # Initial checks
        if message.author == self.user or message.author.bot: return
        source_channel_id = message.channel.id
        if source_channel_id not in self.channel_language_map: return
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only': return
        if source_channel_type not in ['standard', 'read_only']:
            logger.warning(f"Message received from channel {source_channel_id} with unknown type '{source_channel_type}'. Treating as readable.")
        if not message.content and not message.attachments: return
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language:
            logger.warning(f"Internal error: Could not find language for channel ID {source_channel_id}.")
            return

        # Determine writable target channels
        writable_target_channels = defaultdict(list)
        target_language_codes = []
        for lang_code, channel_ids in self.language_channel_map.items():
            if lang_code == source_language: continue
            has_writable_channel_for_lang = False
            for channel_id in channel_ids:
                channel_type = self.channel_type_map.get(channel_id)
                if channel_type in ['standard', 'write_only']:
                    writable_target_channels[lang_code].append(channel_id)
                    has_writable_channel_for_lang = True
            if has_writable_channel_for_lang: target_language_codes.append(lang_code)

        if not writable_target_channels:
            logger.info(f"No writable target channels for message from {message.channel.name} ({source_language}).")
            return

        text_to_translate = message.content
        if not text_to_translate and message.attachments:
             logger.info(f"Message {message.id} contains only attachments. Syncing links.")
             await self.distribute_translations(message, source_language, {}, writable_target_channels)
             return

        logger.info(f"Detected message from channel '{message.channel.name}' ({source_channel_id}, Type: {source_channel_type}, Lang: {source_language}), Author: {message.author.display_name}")
        log_content = text_to_translate[:100] + ('...' if len(text_to_translate) > 100 else '')
        logger.info(f"Preparing to translate content to languages (for writable channels): {target_language_codes}")
        # Keep this debug log if useful
        # logger.debug(f"Content preview for translation: '{log_content}'")

        # --- Perform Translation ---
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized.")
                raise RuntimeError("OpenAI client not initialized")
            translations = await self.translate_text_with_openai(
                text=text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )
            if translations is None:
                await message.channel.send(f"Sorry {message.author.mention}, translation failed. Check logs.", delete_after=15)
                return
            logger.info(f"Successfully received translation results from API.")
            # Keep this debug log if useful
            # logger.debug(f"Translation results: {translations}")
        # --- Error handling ---
        except openai.AuthenticationError:
            logger.error("OpenAI API Auth Failed.")
            await message.channel.send(f"Sorry {message.author.mention}, auth failed.", delete_after=15)
            return
        except openai.RateLimitError:
            logger.warning("OpenAI API Rate Limit.")
            await message.channel.send(f"Sorry {message.author.mention}, too many requests.", delete_after=15)
            return
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API Connection Error: {e}")
            await message.channel.send(f"Sorry {message.author.mention}, connection error.", delete_after=15)
            return
        except openai.APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            await message.channel.send(f"Sorry {message.author.mention}, API error.", delete_after=15)
            return
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}", exc_info=True)
            await message.channel.send(f"Sorry {message.author.mention}, internal error.", delete_after=15)
            return

        # --- Distribute Translation Results ---
        await self.distribute_translations(message, source_language, translations, writable_target_channels)

    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edit events if message is cached (Fallback)."""
        # This handler is less reliable after restarts due to cache.
        # on_raw_message_edit is the primary handler.
        logger.debug(f"on_message_edit triggered for message ID {after.id} (might be ignored).")
        # Basic checks to prevent processing if handled by raw handler or irrelevant
        if after.author.bot: return
        source_channel_id = after.channel.id
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only' or source_channel_id not in self.channel_language_map: return
        # Content check (only if 'before' is available)
        if before and before.content is not None and before.content == after.content:
             logger.debug(f"[on_message_edit] Content of message {after.id} did not change. Skipping.")
             return
        # Avoid double processing if raw handler likely succeeded (e.g., check a flag if needed)
        logger.info(f"[on_message_edit] Processing edit for message {after.id} (potentially redundant, check raw handler logs).")
        # Consider removing or simplifying this handler if on_raw_message_edit proves sufficient.


    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        """Handles raw message edit events directly from the gateway (Primary Handler)."""
        logger.debug(f"on_raw_message_edit triggered for message ID {payload.message_id}")

        if not payload.channel_id or not payload.message_id:
            logger.debug("Raw edit ignored: Missing channel_id or message_id.")
            return

        # Check Channel Configuration
        source_channel_id = payload.channel_id
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only':
            # logger.debug(f"Raw edit ignored: Channel {source_channel_id} is write_only.") # Can be noisy
            return
        if source_channel_id not in self.channel_language_map:
             # logger.debug(f"Raw edit ignored: Channel {source_channel_id} is not configured.") # Can be noisy
             return

        # Extract Edited Content
        new_content = payload.data.get('content', None)
        if new_content is None:
            # logger.debug(f"Raw edit ignored: No 'content' field for message {payload.message_id}.") # Can be noisy
            return

        # Check if Author is Bot
        author_data = payload.data.get('author', {})
        if author_data.get('bot', False):
            # logger.debug(f"Raw edit ignored: Author {author_data.get('id')} is a bot.") # Can be noisy
            return

        # --- Check Database for Tracking ---
        # logger.debug(f"Checking database for message ID: {payload.message_id}...") # Keep if needed
        if not self.db:
            logger.error("Database not available in on_raw_message_edit.")
            return

        tracked_translations = []
        try:
            query = "SELECT target_language, translated_message_id, target_channel_id FROM message_map WHERE original_message_id = ?"
            params = (payload.message_id,)
            # logger.debug(f"Executing DB query: {query} with params: {params}") # Keep if needed
            async with self.db.execute(query, params) as cursor:
                tracked_translations = await cursor.fetchall()
            # logger.debug(f"DB query result for original_message_id {payload.message_id}: {tracked_translations}") # Keep if needed
        except Exception as e:
            logger.error(f"Error querying database for message map of {payload.message_id}: {e}", exc_info=True)
            return

        if not tracked_translations:
            # logger.debug(f"Edited message {payload.message_id} not found in database tracking map. Ignoring raw edit.") # Can be noisy
            return
        # --- End Database Check ---

        # --- Proceed with Re-translation and Update ---
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language:
            logger.warning(f"Internal error: Could not find language for tracked message {payload.message_id}.")
            return

        logger.info(f"Detected edit via raw event in tracked message {payload.message_id} in channel {source_channel_id} ({source_language}).")

        target_language_codes = [row[0] for row in tracked_translations]
        if not target_language_codes:
            logger.warning(f"No target languages found in DB map for edited message {payload.message_id}.")
            return

        log_content = new_content[:100] + ('...' if len(new_content) > 100 else '')
        logger.info(f"Re-translating edited content to languages: {target_language_codes}")
        # logger.debug(f"Edited content preview: '{log_content}'") # Keep if needed

        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized.")
                return
            new_translations = await self.translate_text_with_openai(
                text=new_content, # Use content from payload
                source_lang=source_language,
                target_langs=target_language_codes
            )
            if new_translations is None:
                logger.error(f"Failed to re-translate content for edited message {payload.message_id}.")
                return
            logger.info(f"Successfully received updated translation results for message {payload.message_id}.")
            # logger.debug(f"Updated translations: {new_translations}") # Keep if needed
        except Exception as e:
            logger.error(f"Unexpected error during re-translation for message {payload.message_id}: {e}", exc_info=True)
            return

        # Update the messages using the fetched tracking info
        await self.update_translated_messages(payload.message_id, new_translations, tracked_translations)


    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """Translates text using OpenAI library."""
        if not self.openai_client: logger.error("OpenAI client not initialized."); return None
        model_name = self.settings.get('translation_model', 'google/gemma-3-27b-it')
        system_prompt = f"""You are an expert multilingual translator, you will correctly handle the meaning and context of clearly defined idioms, and you will not add any content that is not present in the original text. Translate the user's text from {source_lang} into the following languages: {', '.join(target_langs)}.
Respond ONLY with a valid JSON object containing the translations. The JSON object should have language codes (exactly as provided: {', '.join(target_langs)}) as keys and the corresponding translated text as string values.
Example format for targets {target_langs}: {{ "{target_langs[0]}": "translation for {target_langs[0]}", "{target_langs[1]}": "translation for {target_langs[1]}" }}
Ensure the output is nothing but the JSON object. Do not include any explanations, markdown formatting around the JSON, or introductory text. Preserve original formatting like markdown (e.g., bold, italics) within the translated strings where appropriate, but prioritize accurate translation. If the input text is empty or contains only whitespace, return an empty JSON object {{}}.
Target languages: {target_langs}"""
        user_prompt = text if text else ""
        content_str = None
        try:
            # logger.debug(f"Sending request to OpenRouter (via OpenAI lib)...") # Can be noisy
            # logger.debug(f"Model: {model_name}, Target Languages: {target_langs}") # Can be noisy
            response = await self.openai_client.chat.completions.create(
                model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}, temperature=0.5, max_tokens=1500 )
            content_str = response.choices[0].message.content
            # logger.debug(f"API response content (raw): {content_str[:500]}...") # Keep if needed
            if not content_str: logger.error("API response successful, but content is empty."); return None
            # Clean Markdown
            cleaned_content_str = content_str.strip()
            if cleaned_content_str.startswith("```json"): cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            if cleaned_content_str.startswith("```"): cleaned_content_str = cleaned_content_str[len("```"):].strip()
            if cleaned_content_str.endswith("```"): cleaned_content_str = cleaned_content_str[:-len("```")].strip()
            if not cleaned_content_str: logger.error("Content is empty after cleaning Markdown symbols."); logger.debug(f"Original content_str: {content_str}"); return None
            # logger.debug(f"Cleaned content ready for parsing: {cleaned_content_str[:500]}...") # Keep if needed
            translations = json.loads(cleaned_content_str)
            if not isinstance(translations, dict): logger.error(f"Parsed content is not a valid JSON object: {translations}"); return None
            # Fill missing
            missing_langs = [lang for lang in target_langs if lang not in translations]
            if missing_langs:
                logger.warning(f"API response missing translations for some languages: {missing_langs}")
                for lang in missing_langs: translations[lang] = f"[{lang.upper()} translation missing]"
            # Filter extra
            filtered_translations = {k: v for k, v in translations.items() if k in target_langs}
            return filtered_translations
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response JSON: {e}")
            original_content_to_log = content_str if content_str is not None else "[Could not get raw response content]"
            logger.error(f"Raw response content: {original_content_to_log}")
            return None
        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"Error processing API response structure: {e}")
             response_to_log = response if 'response' in locals() else "[Could not get response object]"
             logger.debug(f"Full API response object: {response_to_log}") # Keep this debug log
             return None
        except Exception as e:
            logger.error(f"Unexpected error during translation using OpenAI library: {e}", exc_info=True)
            return None


    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, writable_target_map: dict):
        """
        Distribute translations and store mappings in the database.
        """
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []
        lang_codes_in_order = list(translations.keys()) if translations else list(writable_target_map.keys())

        # Create base embed
        base_embed = discord.Embed(color=discord.Color.blue())
        base_embed.set_author(name=f"{original_author.display_name} (from #{original_channel.name})", icon_url=original_author.display_avatar.url if original_author.display_avatar else None)
        base_embed.add_field(name="Original Message", value=f"[Click here to view]({original_message.jump_url})", inline=False)
        if original_message.attachments:
            attachment_links = [f"[{att.filename}]({att.url})" for i, att in enumerate(original_message.attachments) if i < 5]
            if len(original_message.attachments) > 5: attachment_links.append("...")
            if attachment_links: base_embed.add_field(name=f"Attachments ({len(original_message.attachments)})", value="\n".join(attachment_links), inline=False)

        async def _send_translation(target_channel_id, lang_code, translated_text):
            sent_msg_id = None
            try:
                target_channel = self.get_channel(target_channel_id)
                channel_type = self.channel_type_map.get(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel) and channel_type in ['standard', 'write_only']:
                    embed_to_send = base_embed.copy()
                    embed_to_send.description = translated_text if translated_text and translated_text.strip() else "*(No text content or translation result is empty)*"
                    embed_to_send.set_footer(text=f"Original ({source_language.upper()}) → Target ({lang_code.upper()})")
                    sent_msg = await target_channel.send(embed=embed_to_send)
                    sent_msg_id = sent_msg.id
                    logger.info(f"Sent translation ({lang_code}) to {channel_type} channel '{target_channel.name}' ({target_channel_id}) - Msg ID: {sent_msg_id}")
                elif not target_channel or not isinstance(target_channel, discord.TextChannel): logger.warning(f"Could not find target channel ID {target_channel_id}.")
                elif channel_type == 'read_only': logger.warning(f"Attempted send to read_only channel {target_channel_id}. Skipping.")
            except discord.errors.Forbidden: logger.error(f"Permission denied: Cannot send message to channel ID {target_channel_id}.")
            except discord.errors.HTTPException as e: logger.error(f"HTTP error sending message to channel ID {target_channel_id} (Status: {e.status}): {e.text}")
            except Exception as e: logger.error(f"Unexpected error distributing translation to channel ID {target_channel_id}: {e}", exc_info=True)
            return {'msg_id': sent_msg_id, 'channel_id': target_channel_id} if sent_msg_id else None

        # Create send tasks
        if translations:
            for lang_code, translated_text in translations.items():
                if lang_code in writable_target_map:
                    target_channel_id = writable_target_map[lang_code][0]
                    distribution_tasks.append(_send_translation(target_channel_id, lang_code, translated_text))
        elif not translations and original_message.attachments:
             logger.info(f"Syncing attachment links only to all writable target channels...")
             for lang_code, target_channel_ids in writable_target_map.items():
                 target_channel_id = target_channel_ids[0]
                 distribution_tasks.append(_send_translation(target_channel_id, lang_code, ""))
        else: logger.warning("No translation results and no attachments to distribute.")

        # Run tasks and collect results
        if distribution_tasks:
            results = await asyncio.gather(*distribution_tasks)
            # Store mapping in Database
            if not self.db:
                logger.error("Database not available, cannot store message map.")
                return

            insert_data_list = []
            for i, result_detail in enumerate(results):
                if result_detail and result_detail.get('msg_id'):
                    lang_code = lang_codes_in_order[i]
                    translated_msg_id = result_detail['msg_id']
                    target_channel_id = result_detail['channel_id']
                    insert_data_list.append((
                        original_message.id,
                        lang_code,
                        translated_msg_id,
                        target_channel_id
                    ))
                    # logger.debug(f"Prepared DB insert for original {original_message.id}, lang {lang_code}, translated {translated_msg_id}") # Can be noisy

            if insert_data_list:
                try:
                    await self.db.executemany(
                        "INSERT OR REPLACE INTO message_map (original_message_id, target_language, translated_message_id, target_channel_id) VALUES (?, ?, ?, ?)",
                        insert_data_list
                    )
                    await self.db.commit()
                    logger.info(f"Successfully stored/updated {len(insert_data_list)} mappings in database for original message {original_message.id}.")
                except Exception as e:
                    logger.error(f"Failed to store message mappings in database for original message {original_message.id}: {e}", exc_info=True)
                    try: await self.db.rollback()
                    except Exception as e_roll: logger.error(f"Database rollback failed: {e_roll}")

            logger.info(f"Finished distributing messages for original message {original_message.id}.")


    async def update_translated_messages(self, original_message_id: int, new_translations: dict, tracked_translations: list):
        """
        Finds and edits translated messages based on data fetched from the database.
        """
        update_tasks = []

        async def _edit_translation(lang_code, translated_message_id: int, target_channel_id: int, new_text):
            try:
                target_channel = self.get_channel(target_channel_id)
                if not target_channel:
                     logger.warning(f"Could not find target channel {target_channel_id} for message {translated_message_id} (lang {lang_code}). Removing entry from map.")
                     if self.db:
                         try:
                             await self.db.execute("DELETE FROM message_map WHERE original_message_id = ? AND target_language = ?", (original_message_id, lang_code))
                             await self.db.commit()
                             logger.info(f"Removed missing channel entry from DB for original {original_message_id}, lang {lang_code}")
                         except Exception as e_del: logger.error(f"Failed to remove DB entry for missing channel: {e_del}")
                     return

                translated_message = await target_channel.fetch_message(translated_message_id)

                if translated_message.embeds:
                    original_embed = translated_message.embeds[0]
                    new_embed = original_embed.copy()
                    new_embed.description = new_text if new_text and new_text.strip() else "*(No text content or translation result is empty)*"
                    await translated_message.edit(embed=new_embed)
                    logger.info(f"Successfully edited translated message {translated_message_id} in channel '{target_channel.name}' for language {lang_code}.")
                else:
                    logger.warning(f"Translated message {translated_message_id} has no embed to edit.")

            except discord.NotFound:
                logger.warning(f"Translated message {translated_message_id} not found in channel {target_channel_id} (likely deleted). Removing from map.")
                if self.db:
                    try:
                        await self.db.execute("DELETE FROM message_map WHERE original_message_id = ? AND target_language = ?", (original_message_id, lang_code))
                        await self.db.commit()
                        logger.info(f"Removed deleted message entry from DB for original {original_message_id}, lang {lang_code}")
                    except Exception as e_del: logger.error(f"Failed to remove DB entry for deleted message: {e_del}")
            except discord.Forbidden:
                logger.error(f"Permission error editing message {translated_message_id} in channel {target_channel_id}.")
            except Exception as e:
                logger.error(f"Unexpected error editing translated message {translated_message_id}: {e}", exc_info=True)

        # Create tasks to edit each translated message
        for lang, msg_id, chan_id in tracked_translations:
            if lang in new_translations:
                update_tasks.append(
                    _edit_translation(lang, msg_id, chan_id, new_translations[lang])
                )
            else:
                 logger.warning(f"New translation missing for language {lang} during update for original message {original_message_id}.")

        # Run update tasks
        if update_tasks:
            await asyncio.gather(*update_tasks)
            logger.info(f"Finished updating translations for original message {original_message_id}.")


# --- Main Entry Point ---
if __name__ == "__main__":
    # Read config temporarily
    temp_config = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: temp_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.critical(f"Error: Configuration file {CONFIG_FILE} not found. Please create it first.")
        try: TranslationBot(intents=discord.Intents.default(), config_path=CONFIG_FILE).create_example_config()
        except Exception as e_create: logger.error(f"Failed to create example config: {e_create}")
        exit(1)
    except yaml.YAMLError as e: logger.critical(f"Error: Failed to parse configuration file {CONFIG_FILE}: {e}"); exit(1)

    # Validate config
    api_keys = temp_config.get('api_keys', {})
    channels_config = temp_config.get('channels', {})
    discord_token = os.getenv('DISCORD_BOT_TOKEN') or api_keys.get('discord_token')
    openrouter_key = api_keys.get('openrouter_key')
    config_valid = True
    if not discord_token or discord_token == 'YOUR_DISCORD_BOT_TOKEN': logger.critical("Error: Valid Discord Bot Token not configured."); config_valid = False
    if not openrouter_key or openrouter_key == 'YOUR_OPENROUTER_API_KEY': logger.critical("Error: Valid OpenRouter API Key not configured."); config_valid = False
    if not isinstance(channels_config, dict) or not channels_config: logger.critical("Error: 'channels' section missing, invalid, or empty in config.yaml."); config_valid = False
    elif not any(langs for type_data in channels_config.values() if isinstance(type_data, dict) for langs in type_data.values()): logger.critical("Error: No channels defined under standard, read_only, or write_only in config.yaml."); config_valid = False
    if not config_valid: exit(1)

    # Set up Intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    logger.info("Configuring Intents...")
    if not intents.message_content:
        logger.critical("CRITICAL: Message Content Intent is NOT enabled in the client code or Developer Portal!")
    else:
         logger.info("Message Content Intent is enabled.")

    # Initialize and run Bot
    bot = TranslationBot(intents=intents, config_path=CONFIG_FILE)
    logger.info("Starting Bot...")
    try:
        bot.run(discord_token, log_handler=None)
    except discord.LoginFailure: logger.critical("Login Failed: Invalid Discord Token.")
    except discord.PrivilegedIntentsRequired: logger.critical("Login Failed: Missing necessary Privileged Intents.")
    except Exception as e: logger.critical(f"Critical error during Bot startup: {e}", exc_info=True)
    finally:
        pass
