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

# --- Configuration ---
CONFIG_FILE = 'config.yaml'
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('TranslationBot')

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Core Bot Class ---
class TranslationBot(discord.Client):
    """
    Translation bot class inheriting from discord.Client.
    Includes message edit tracking and channel type support (standard, read_only, write_only).
    """
    def __init__(self, *, intents: discord.Intents, config_path: str):
        super().__init__(intents=intents)
        self.config_path = config_path
        # --- Modified Mappings ---
        self.channel_language_map = {} # {channel_id: lang_code} (All configured channels)
        self.channel_type_map = {}     # {channel_id: type} ('standard', 'read_only', 'write_only')
        self.language_channel_map = defaultdict(list) # {lang_code: [channel_id, ...]} (All configured channels for that lang)
        # --- End Modified Mappings ---
        self.api_keys = {}
        self.settings = {
            'translation_model': 'google/gemini-flash',
            'max_retries': 2,
            'retry_delay': 3
        }
        self.openai_client: AsyncOpenAI | None = None
        self.message_map = defaultdict(dict) # {original_msg_id: {target_lang: translated_msg_id}}
        self.message_map_max_size = 10000

    async def setup_hook(self) -> None:
        self.load_config()
        # Log listening channel count based on readable channels
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
        await super().close()
        logger.info("Bot is shutting down.")

    def load_config(self):
        """Load settings from the YAML file, parsing new channel structure."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if not config_data:
                    logger.error(f"Configuration file '{self.config_path}' is empty.")
                    self.create_example_config()
                    logger.critical("Example configuration file created. Please fill in your information and restart the Bot.")
                    return

                # --- Modified Channel Loading ---
                self.channel_language_map = {}
                self.channel_type_map = {}
                self.language_channel_map = defaultdict(list)
                total_channels_loaded = 0

                if 'channels' in config_data and isinstance(config_data['channels'], dict):
                    channel_configs = config_data['channels']
                    # Process each channel type section
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
                                channel_ids = [channel_ids] # Allow single ID or list

                            for channel_id in channel_ids:
                                try:
                                    ch_id_int = int(channel_id)
                                    # Check for duplicates across all types
                                    if ch_id_int in self.channel_language_map:
                                        logger.warning(f"Channel ID {ch_id_int} was already configured as language '{self.channel_language_map[ch_id_int]}' (type: {self.channel_type_map.get(ch_id_int, 'unknown')}). Overwriting with language '{lang_code}' (type: {channel_type}).")
                                    # Store mappings
                                    self.channel_language_map[ch_id_int] = lang_code
                                    self.channel_type_map[ch_id_int] = channel_type
                                    self.language_channel_map[lang_code].append(ch_id_int)
                                    total_channels_loaded += 1
                                    logger.debug(f"  Loaded Channel ID: {ch_id_int}, Lang: {lang_code}, Type: {channel_type}")
                                except ValueError:
                                    logger.warning(f"Invalid channel ID '{channel_id}' found under language '{lang_code}' (type: {channel_type}). Skipping.")
                    logger.info(f"Successfully loaded {total_channels_loaded} channel configurations across all types.")
                    # Log summary for debugging
                    logger.debug(f"Final channel_language_map: {self.channel_language_map}")
                    logger.debug(f"Final channel_type_map: {self.channel_type_map}")
                    logger.debug(f"Final language_channel_map: {dict(self.language_channel_map)}") # Convert defaultdict for cleaner log
                else:
                    logger.warning(f"Missing or invalid 'channels' section in configuration file '{self.config_path}'. No channels loaded.")
                # --- End Modified Channel Loading ---

                # Load API keys (same as before)
                if 'api_keys' in config_data and isinstance(config_data['api_keys'], dict):
                    self.api_keys = config_data['api_keys']
                    logger.info("Successfully loaded API key settings.")
                else: logger.warning(f"Missing or invalid 'api_keys' section in configuration file '{self.config_path}'.")

                # Load other settings (same as before)
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
        """Create an example configuration file with the new structure."""
        default_config = {
            'api_keys': {
                'discord_token': 'YOUR_DISCORD_BOT_TOKEN',
                'openrouter_key': 'YOUR_OPENROUTER_API_KEY'
            },
            'channels': {
                'standard': {
                    'en': [123456789012345678], # Example English standard channel
                    'zh-TW': [987654321098765432] # Example Chinese Traditional standard channel
                },
                'read_only': {
                    'ja': [111111111111111111] # Example Japanese read-only channel
                },
                'write_only': {
                    'ko': [222222222222222222] # Example Korean write-only channel
                }
            },
            'settings': {
                'translation_model': 'google/gemini-flash',
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
        # Ignore messages from self or other bots
        if message.author == self.user or message.author.bot: return

        source_channel_id = message.channel.id
        # Ignore messages not from *any* configured channel
        if source_channel_id not in self.channel_language_map: return

        # --- Added: Check source channel type ---
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only':
            logger.debug(f"Ignoring message from write_only channel {message.channel.name} ({source_channel_id}).")
            return
        # If type is unknown (shouldn't happen with proper loading), treat as standard or log warning
        if source_channel_type not in ['standard', 'read_only']:
             logger.warning(f"Message received from channel {source_channel_id} with unknown or missing type '{source_channel_type}'. Treating as readable.")
             # Decide how to handle: treat as standard/read_only or ignore? Let's treat as readable.

        # Ignore messages with no content and no attachments
        if not message.content and not message.attachments:
            logger.debug(f"Message from channel {message.channel.name} has empty content and no attachments. Ignoring.")
            return

        # Get source language
        source_language = self.channel_language_map.get(source_channel_id)
        # This check is technically redundant now but good for safety
        if not source_language:
            logger.warning(f"Internal error: Could not find language setting for channel ID {source_channel_id}.")
            return

        # --- Modified: Determine writable target channels ---
        writable_target_channels = defaultdict(list) # {lang_code: [writable_channel_id, ...]}
        target_language_codes = []

        for lang_code, channel_ids in self.language_channel_map.items():
            if lang_code == source_language:
                continue # Skip source language

            has_writable_channel_for_lang = False
            for channel_id in channel_ids:
                channel_type = self.channel_type_map.get(channel_id)
                # Only add standard and write_only channels as targets
                if channel_type in ['standard', 'write_only']:
                    writable_target_channels[lang_code].append(channel_id)
                    has_writable_channel_for_lang = True

            if has_writable_channel_for_lang:
                target_language_codes.append(lang_code)
        # --- End Modified Target Channel Determination ---

        if not writable_target_channels:
            logger.info(f"No writable target channels found for message from {message.channel.name} ({source_language}).")
            return

        text_to_translate = message.content
        if not text_to_translate and message.attachments:
             logger.info(f"Message {message.id} contains only attachments. Syncing attachment links only.")
             # Pass the writable target map to distribute
             await self.distribute_translations(message, source_language, {}, writable_target_channels)
             return

        logger.info(f"Detected message from channel '{message.channel.name}' ({source_channel_id}, Type: {source_channel_type}, Lang: {source_language}), Author: {message.author.display_name}")
        log_content = text_to_translate[:100] + ('...' if len(text_to_translate) > 100 else '')
        logger.info(f"Preparing to translate content to languages (for writable channels): {target_language_codes}")
        logger.debug(f"Content preview for translation: '{log_content}'")

        # --- Perform Translation (logic unchanged) ---
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized. Cannot perform translation.")
                raise RuntimeError("OpenAI client not initialized")

            translations = await self.translate_text_with_openai(
                text=text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes # Translate for all potential target languages
            )

            if translations is None:
                await message.channel.send(f"Sorry {message.author.mention}, there was a problem during translation. Could not complete sync. Please check logs for details.", delete_after=15)
                return

            logger.info(f"Successfully received translation results from API.")
            logger.debug(f"Translation results: {translations}")

        # Error handling (logic unchanged)
        except openai.AuthenticationError:
             logger.error("OpenAI API Authentication Failed: Invalid API Key or insufficient permissions.")
             await message.channel.send(f"Sorry {message.author.mention}, translation service authentication failed.", delete_after=15)
             return
        except openai.RateLimitError:
            logger.warning("OpenAI API Rate Limit Exceeded: Too many requests.")
            await message.channel.send(f"Sorry {message.author.mention}, translation requests are too frequent.", delete_after=15)
            return
        except openai.APIConnectionError as e:
            logger.error(f"Could not connect to OpenAI API: {e}")
            await message.channel.send(f"Sorry {message.author.mention}, could not connect to the translation service.", delete_after=15)
            return
        except openai.APIError as e:
             logger.error(f"OpenAI API returned an error: {e}")
             await message.channel.send(f"Sorry {message.author.mention}, the translation service encountered an error.", delete_after=15)
             return
        except Exception as e:
            logger.error(f"Unexpected error during translation process: {e}", exc_info=True)
            await message.channel.send(f"Sorry {message.author.mention}, an internal error occurred during translation.", delete_after=15)
            return

        # --- Distribute Translation Results ---
        # Pass the map of *writable* target channels
        await self.distribute_translations(message, source_language, translations, writable_target_channels)

    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edit events, considering channel types."""
        # Ignore edits from bots or if content hasn't changed
        if after.author.bot or before.content == after.content:
            return

        source_channel_id = after.channel.id
        # --- Modified: Check if source channel is readable ---
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only':
            # Don't track edits from write_only channels (they shouldn't be in map anyway)
            return
        if source_channel_id not in self.channel_language_map:
            # Also ignore if not a configured channel at all
            return

        # Ignore edits to messages we haven't tracked
        if after.id not in self.message_map:
            logger.debug(f"Edited message {after.id} not found in tracking map. Ignoring edit.")
            return

        # Get original source language
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language:
             logger.warning(f"Internal error: Could not find language for tracked message {after.id} in channel {source_channel_id}.")
             return

        logger.info(f"Detected edit in tracked message {after.id} in channel '{after.channel.name}' ({source_language}).")

        # Determine target languages based on existing map for this message
        target_language_codes = list(self.message_map[after.id].keys())
        if not target_language_codes:
            logger.warning(f"No target languages found in map for edited message {after.id}. Cannot update translations.")
            return

        new_text_to_translate = after.content
        log_content = new_text_to_translate[:100] + ('...' if len(new_text_to_translate) > 100 else '')
        logger.info(f"Re-translating edited content to languages: {target_language_codes}")
        logger.debug(f"Edited content preview: '{log_content}'")

        # --- Perform Re-Translation (logic unchanged) ---
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized. Cannot perform re-translation.")
                return

            new_translations = await self.translate_text_with_openai(
                text=new_text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )

            if new_translations is None:
                logger.error(f"Failed to re-translate content for edited message {after.id}.")
                return

            logger.info(f"Successfully received updated translation results for message {after.id}.")
            logger.debug(f"Updated translations: {new_translations}")

        except Exception as e:
            logger.error(f"Unexpected error during re-translation process for message {after.id}: {e}", exc_info=True)
            return

        # --- Update Translated Messages ---
        await self.update_translated_messages(after.id, new_translations)


    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """
        Translates text into multiple target languages using the openai library.
        (Logic remains the same)
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized.")
            return None
        model_name = self.settings.get('translation_model', 'google/gemini-flash')
        system_prompt = f"""You are an expert multilingual translator. Translate the user's text from {source_lang} into the following languages: {', '.join(target_langs)}.
Respond ONLY with a valid JSON object containing the translations. The JSON object should have language codes (exactly as provided: {', '.join(target_langs)}) as keys and the corresponding translated text as string values.
Example format for targets {target_langs}: {{ "{target_langs[0]}": "translation for {target_langs[0]}", "{target_langs[1]}": "translation for {target_langs[1]}" }}
Ensure the output is nothing but the JSON object. Do not include any explanations, markdown formatting around the JSON, or introductory text. Preserve original formatting like markdown (e.g., bold, italics) within the translated strings where appropriate, but prioritize accurate translation. If the input text is empty or contains only whitespace, return an empty JSON object {{}}.
Target languages: {target_langs}"""
        user_prompt = text if text else ""
        content_str = None
        try:
            logger.debug(f"Sending request to OpenRouter (via OpenAI lib)...")
            logger.debug(f"Model: {model_name}, Target Languages: {target_langs}")
            response = await self.openai_client.chat.completions.create(
                model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}, temperature=0.5, max_tokens=1500 )
            content_str = response.choices[0].message.content
            logger.debug(f"API response content (raw): {content_str[:500]}...")
            if not content_str: logger.error("API response successful, but content is empty."); return None
            cleaned_content_str = content_str.strip()
            if cleaned_content_str.startswith("```json"): cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            if cleaned_content_str.startswith("```"): cleaned_content_str = cleaned_content_str[len("```"):].strip()
            if cleaned_content_str.endswith("```"): cleaned_content_str = cleaned_content_str[:-len("```")].strip()
            if not cleaned_content_str: logger.error("Content is empty after cleaning Markdown symbols."); logger.debug(f"Original content_str: {content_str}"); return None
            logger.debug(f"Cleaned content ready for parsing: {cleaned_content_str[:500]}...")
            translations = json.loads(cleaned_content_str)
            if not isinstance(translations, dict): logger.error(f"Parsed content is not a valid JSON object: {translations}"); return None
            missing_langs = [lang for lang in target_langs if lang not in translations]
            if missing_langs:
                logger.warning(f"API response missing translations for some languages: {missing_langs}")
                for lang in missing_langs: translations[lang] = f"[{lang.upper()} translation missing]"
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
             logger.debug(f"Full API response object: {response_to_log}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error during translation using OpenAI library: {e}", exc_info=True)
            return None

    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, writable_target_map: dict):
        """
        Distribute translation results ONLY to writable target channels and store message mappings.
        Args:
            writable_target_map (dict): Map of {lang_code: [writable_channel_id, ...]}
        """
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []
        # --- Modified: Use language codes from the writable map ---
        lang_codes_in_order = list(translations.keys()) if translations else list(writable_target_map.keys())
        sent_message_details = {} # {lang_code: {'msg_id': id, 'channel_id': id}}

        # Create a base embed
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
                # Double check type here for safety, although map should be pre-filtered
                channel_type = self.channel_type_map.get(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel) and channel_type in ['standard', 'write_only']:
                    embed_to_send = base_embed.copy()
                    embed_to_send.description = translated_text if translated_text and translated_text.strip() else "*(No text content or translation result is empty)*"
                    embed_to_send.set_footer(text=f"Original ({source_language.upper()}) → Target ({lang_code.upper()})")
                    sent_msg = await target_channel.send(embed=embed_to_send)
                    sent_msg_id = sent_msg.id
                    logger.info(f"Sent translation ({lang_code}) to {channel_type} channel '{target_channel.name}' ({target_channel_id}) - Msg ID: {sent_msg_id}")
                elif not target_channel or not isinstance(target_channel, discord.TextChannel):
                    logger.warning(f"Could not find target channel ID {target_channel_id} or it's not a valid text channel.")
                elif channel_type == 'read_only':
                     logger.warning(f"Attempted to send translation to read_only channel {target_channel_id}. Skipping (this indicates a logic error).") # Should not happen
            except discord.errors.Forbidden: logger.error(f"Permission denied: Cannot send message to channel ID {target_channel_id}.")
            except discord.errors.HTTPException as e: logger.error(f"HTTP error sending message to channel ID {target_channel_id} (Status: {e.status}): {e.text}")
            except Exception as e: logger.error(f"Unexpected error distributing translation to channel ID {target_channel_id}: {e}", exc_info=True)
            # Return dict with message ID and channel ID for mapping
            return {'msg_id': sent_msg_id, 'channel_id': target_channel_id} if sent_msg_id else None

        # Create send tasks using the writable_target_map
        if translations:
            for lang_code, translated_text in translations.items():
                if lang_code in writable_target_map:
                    # Send to the first writable channel for this language
                    target_channel_id = writable_target_map[lang_code][0]
                    distribution_tasks.append(_send_translation(target_channel_id, lang_code, translated_text))
                # Don't warn here, as translations dict might contain languages without writable channels
        elif not translations and original_message.attachments:
             logger.info(f"Syncing attachment links only to all writable target channels...")
             for lang_code, target_channel_ids in writable_target_map.items():
                 target_channel_id = target_channel_ids[0]
                 distribution_tasks.append(_send_translation(target_channel_id, lang_code, ""))
        else:
             logger.warning("No translation results and no attachments to distribute.")

        # Run tasks and collect results (dictionaries with msg_id and channel_id)
        if distribution_tasks:
            results = await asyncio.gather(*distribution_tasks)
            # Store the mapping {original_id: {lang_code: {'msg_id': id, 'channel_id': id}}}
            successful_sends_map = {}
            for i, result_detail in enumerate(results):
                if result_detail and result_detail.get('msg_id'): # Check if sending was successful
                    lang_code = lang_codes_in_order[i]
                    successful_sends_map[lang_code] = result_detail # Store {'msg_id': id, 'channel_id': id}

            if successful_sends_map:
                 self.message_map[original_message.id] = successful_sends_map
                 logger.info(f"Stored mapping for original message {original_message.id}: {successful_sends_map}")
                 # Prune map if needed
                 if len(self.message_map) > self.message_map_max_size:
                     oldest_key = next(iter(self.message_map))
                     del self.message_map[oldest_key]
                     logger.info(f"Message map size exceeded {self.message_map_max_size}. Removed oldest entry: {oldest_key}")

            logger.info(f"Finished distributing messages for original message {original_message.id}.")


    async def update_translated_messages(self, original_message_id: int, new_translations: dict):
        """
        Finds and edits translated messages based on the stored mapping,
        respecting channel types.
        """
        if original_message_id not in self.message_map:
            logger.warning(f"Cannot update translations for {original_message_id}: No mapping found.")
            return

        translation_map = self.message_map[original_message_id] # {lang_code: {'msg_id': id, 'channel_id': id}}
        update_tasks = []

        async def _edit_translation(lang_code, message_detail: dict, new_text):
            translated_message_id = message_detail.get('msg_id')
            target_channel_id = message_detail.get('channel_id') # Get channel ID from map

            if not translated_message_id or not target_channel_id:
                 logger.warning(f"Invalid message detail in map for lang {lang_code}, original msg {original_message_id}. Skipping update.")
                 return

            try:
                # Fetch channel using stored ID
                target_channel = self.get_channel(target_channel_id)
                if not target_channel:
                     logger.warning(f"Could not find target channel {target_channel_id} for message {translated_message_id} (lang {lang_code}). Removing entry from map.")
                     if lang_code in self.message_map[original_message_id]:
                         del self.message_map[original_message_id][lang_code]
                     return

                # Fetch the message directly from the known channel
                translated_message = await target_channel.fetch_message(translated_message_id)

                # Edit the embed
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
                if lang_code in self.message_map[original_message_id]:
                     del self.message_map[original_message_id][lang_code]
            except discord.Forbidden:
                logger.error(f"Permission error editing message {translated_message_id} in channel {target_channel_id}.")
            except Exception as e:
                logger.error(f"Unexpected error editing translated message {translated_message_id}: {e}", exc_info=True)

        # Create tasks to edit each translated message
        for lang_code, message_detail in translation_map.items():
            if lang_code in new_translations:
                update_tasks.append(
                    _edit_translation(lang_code, message_detail, new_translations[lang_code])
                )
            else:
                 logger.warning(f"New translation missing for language {lang_code} during update for original message {original_message_id}.")

        # Run update tasks concurrently
        if update_tasks:
            await asyncio.gather(*update_tasks)
            logger.info(f"Finished updating translations for original message {original_message_id}.")


# --- Main Entry Point (logic unchanged) ---
if __name__ == "__main__":
    temp_config = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: temp_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.critical(f"Error: Configuration file {CONFIG_FILE} not found. Please create it first.")
        # Attempt to create example, but exit anyway as token is missing
        try:
             TranslationBot(intents=discord.Intents.default(), config_path=CONFIG_FILE).create_example_config()
        except Exception as e_create:
             logger.error(f"Failed to create example config: {e_create}")
        exit(1)
    except yaml.YAMLError as e:
         logger.critical(f"Error: Failed to parse configuration file {CONFIG_FILE}: {e}")
         exit(1)

    api_keys = temp_config.get('api_keys', {})
    channels_config = temp_config.get('channels', {}) # Check if channels section exists
    discord_token = os.getenv('DISCORD_BOT_TOKEN') or api_keys.get('discord_token')
    openrouter_key = api_keys.get('openrouter_key')
    config_valid = True
    if not discord_token or discord_token == 'YOUR_DISCORD_BOT_TOKEN':
        logger.critical("Error: Valid Discord Bot Token not configured."); config_valid = False
    if not openrouter_key or openrouter_key == 'YOUR_OPENROUTER_API_KEY':
        logger.critical("Error: Valid OpenRouter API Key not configured."); config_valid = False
    # Check if the main 'channels' key exists and is a dictionary
    if not isinstance(channels_config, dict) or not channels_config:
        logger.critical("Error: 'channels' section missing, invalid, or empty in config.yaml."); config_valid = False
    # Further check if at least one channel is defined under any type
    elif not any(langs for type_data in channels_config.values() if isinstance(type_data, dict) for langs in type_data.values()):
         logger.critical("Error: No channels defined under standard, read_only, or write_only in config.yaml."); config_valid = False

    if not config_valid: exit(1)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    logger.info("Configuring Intents...")
    if not intents.message_content: logger.warning("Warning: Message Content Intent is not enabled.")

    bot = TranslationBot(intents=intents, config_path=CONFIG_FILE)
    logger.info("Starting Bot...")
    try:
        bot.run(discord_token, log_handler=None)
    except discord.LoginFailure: logger.critical("Login Failed: Invalid Discord Token.")
    except discord.PrivilegedIntentsRequired: logger.critical("Login Failed: Missing necessary Privileged Intents.")
    except Exception as e: logger.critical(f"Critical error during Bot startup: {e}", exc_info=True)
    finally: pass
