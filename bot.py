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
import aiohttp # Added for Ollama

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
    and channel type support. Features improved message format and fallback translation.
    """
    def __init__(self, *, intents: discord.Intents, config_path: str):
        if not intents.messages:
             logger.warning("discord.Intents.messages is not enabled. Bot might miss message events.")
        if not intents.message_content:
             logger.warning("discord.Intents.message_content is not enabled. Bot cannot read message content.")

        super().__init__(intents=intents)
        self.config_path = config_path
        self.channel_language_map = {}
        self.channel_type_map = {}
        self.language_channel_map = defaultdict(list)
        self.api_keys = {} # Still used for discord_token
        self.settings = { # General settings, model moved to service config
            'max_retries': 2,
            'retry_delay': 3,
            'language_separation_threshold': 1200 # Default value, will be overwritten by config
        }
        # New attributes for translation services
        self.translation_services = {}  # Stores service configurations from config.yaml
        self.translation_clients = {}   # Stores initialized API clients (AsyncOpenAI instances)
        self.enable_remote_fallback = False
        self.enable_ollama_fallback = False
        self.ollama_config = {}
        self.db: aiosqlite.Connection | None = None

    async def setup_hook(self) -> None:
        """Initialize database and load config."""
        await self.initialize_database()
        self.load_config() # Loads self.translation_services, self.enable_*, self.ollama_config etc.

        readable_channels = [cid for cid, type in self.channel_type_map.items() if type in ['standard', 'read_only']]
        logger.info(f'Configuration loaded. Listening for messages in {len(readable_channels)} channels.')

        # --- Initialize Translation Clients ---
        self.translation_clients = {} # Ensure it's clean before init
        initialized_clients = 0

        # Initialize primary service
        primary_config = self.translation_services.get('primary')
        if primary_config:
            if self._init_openai_client('primary', primary_config):
                initialized_clients += 1
        else:
            logger.warning("No primary translation service configured.")

        # Initialize remote fallback services
        if self.enable_remote_fallback:
            fallback_configs = self.translation_services.get('remote_fallbacks', [])
            if fallback_configs:
                logger.info(f"Initializing {len(fallback_configs)} remote fallback service(s)...")
                for i, fallback_config in enumerate(fallback_configs):
                    service_id = f"fallback_{i}"
                    if self._init_openai_client(service_id, fallback_config):
                        initialized_clients += 1
            else:
                logger.warning("Remote fallback enabled, but no fallback configurations found.")
        else:
             logger.info("Remote fallback services are disabled.")

        # Note: Ollama doesn't need an AsyncOpenAI client, its config is used directly in its method.
        if self.enable_ollama_fallback:
            logger.info(f"Ollama fallback enabled (Model: {self.ollama_config.get('model', 'N/A')}). Client is created on demand.")
        else:
            logger.info("Ollama fallback is disabled.")

        if initialized_clients == 0 and not self.enable_ollama_fallback:
             logger.error("CRITICAL: No translation services (primary, remote, or Ollama) were successfully configured or initialized. Bot may not translate.")
        elif initialized_clients == 0 and self.enable_ollama_fallback:
             logger.warning("No remote translation services initialized. Relying solely on Ollama fallback.")
        else:
             logger.info(f"Successfully initialized {initialized_clients} remote translation client(s).")

        # Remove old client attribute reference if it exists (optional cleanup)
        if hasattr(self, 'openai_client'):
            del self.openai_client
            logger.debug("Removed legacy 'openai_client' attribute.")


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
                                except ValueError:
                                    logger.warning(f"Invalid channel ID '{channel_id}' found under language '{lang_code}' (type: {channel_type}). Skipping.")
                    logger.info(f"Successfully loaded {total_channels_loaded} channel configurations across all types.")
                else:
                    logger.warning(f"Missing or invalid 'channels' section in configuration file '{self.config_path}'. No channels loaded.")

                # Load API keys
                if 'api_keys' in config_data and isinstance(config_data['api_keys'], dict):
                    self.api_keys = config_data['api_keys']
                    logger.info("Successfully loaded API key settings.")
                else: logger.warning(f"Missing or invalid 'api_keys' section in configuration file '{self.config_path}'.")

                # Load other settings
                if 'settings' in config_data and isinstance(config_data['settings'], dict):
                    # self.settings['translation_model'] = config_data['settings'].get('translation_model', self.settings['translation_model']) # Removed as model is per-service
                    self.settings['max_retries'] = int(config_data['settings'].get('max_retries', self.settings['max_retries']))
                    self.settings['retry_delay'] = int(config_data['settings'].get('retry_delay', self.settings['retry_delay']))
                    self.settings['language_separation_threshold'] = int(config_data['settings'].get('language_separation_threshold', self.settings['language_separation_threshold']))
                    logger.info(f"Successfully loaded general settings (Threshold: {self.settings['language_separation_threshold']}).")
                else: logger.info(f"No 'settings' section found, using default settings (Threshold: {self.settings['language_separation_threshold']}).")

                # Load translation services configuration
                if 'translation_services' in config_data and isinstance(config_data['translation_services'], dict):
                    ts_config = config_data['translation_services']
                    logger.info("Loading translation services configuration...")

                    # Load primary service
                    if 'primary' in ts_config and isinstance(ts_config['primary'], dict):
                        self.translation_services['primary'] = ts_config['primary']
                        logger.info(f"Loaded primary translation service: {ts_config['primary'].get('provider', 'N/A')}")
                    else:
                        logger.warning("Primary translation service configuration ('primary') is missing or invalid.")

                    # Load remote fallback settings
                    self.enable_remote_fallback = ts_config.get('enable_remote_fallback', False)
                    if self.enable_remote_fallback:
                        if 'remote_fallbacks' in ts_config and isinstance(ts_config['remote_fallbacks'], list):
                            self.translation_services['remote_fallbacks'] = ts_config['remote_fallbacks']
                            logger.info(f"Remote fallback enabled. Loaded {len(ts_config['remote_fallbacks'])} fallback configurations.")
                        else:
                            logger.warning("Remote fallback is enabled but 'remote_fallbacks' list is missing or invalid. Disabling remote fallback.")
                            self.enable_remote_fallback = False
                    else:
                        logger.info("Remote fallback is disabled.")

                    # Load Ollama fallback settings
                    self.enable_ollama_fallback = ts_config.get('enable_ollama_fallback', False)
                    if self.enable_ollama_fallback:
                        if 'ollama' in ts_config and isinstance(ts_config['ollama'], dict):
                            self.ollama_config = ts_config['ollama']
                            logger.info(f"Ollama local fallback enabled. Loaded configuration for model: {self.ollama_config.get('model', 'N/A')}")
                        else:
                            logger.warning("Ollama fallback is enabled but 'ollama' configuration is missing or invalid. Disabling Ollama fallback.")
                            self.enable_ollama_fallback = False
                    else:
                        logger.info("Ollama local fallback is disabled.")
                else:
                    logger.warning("Missing or invalid 'translation_services' section in configuration. No translation services loaded.")


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
                'discord_token': 'YOUR_DISCORD_BOT_TOKEN'
                # API keys for translation services are now under 'translation_services'
            },
            'channels': {
                # Keep the existing channel structure example
                'standard': {
                    'en': [123456789012345678], # Example English channel
                    'zh-TW': [987654321098765432] # Example Traditional Chinese channel
                },
                'read_only': {
                    'ja': [111111111111111111] # Example Japanese read-only channel
                },
                'write_only': {
                    'ko': [222222222222222222] # Example Korean write-only channel
                }
            },
            'translation_services': {
                'primary': {
                    'provider': 'openrouter',
                    'base_url': 'https://openrouter.ai/api/v1',
                    'model': 'google/gemma-3-27b-it',
                    'api_key': 'YOUR_OPENROUTER_API_KEY' # Moved here from api_keys
                },
                'enable_remote_fallback': False,
                'remote_fallbacks': [
                    {
                        'provider': 'openai',
                        'base_url': 'https://api.openai.com/v1',
                        'model': 'gpt-4o',
                        'api_key': 'YOUR_OPENAI_API_KEY'
                    },
                    # Add more fallback services here if needed
                    # {
                    #     'provider': 'azure',
                    #     'base_url': 'https://your-resource.openai.azure.com',
                    #     'model': 'gpt-35-turbo', # Or your deployment name
                    #     'api_key': 'YOUR_AZURE_API_KEY',
                    #     'api_version': '2023-05-15',
                    #     'deployment_name': 'your-deployment'
                    # }
                ],
                'enable_ollama_fallback': False,
                'ollama': {
                    'base_url': 'http://localhost:11434', # Corrected: Base URL only, path is added in code
                    'model': 'llama3', # Example model, change if needed
                    'timeout': 30 # Request timeout in seconds
                }
            },
            'settings': {
                # 'translation_model' is now defined per service in 'translation_services'
                'max_retries': 2, # General retry setting (can be overridden per service if needed later)
                'retry_delay': 3 # General retry delay
            }
        }
        try:
            # Ensure the directory exists (though it should in this context)
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)
            logger.info(f"Example configuration file '{self.config_path}' created with the new translation service structure. Please fill in your information.")
        except Exception as e:
            logger.error(f"Error creating example configuration file '{self.config_path}': {e}")


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

        # --- Perform Translation using Fallback System ---
        translations = await self.translate_with_fallback(
            text=text_to_translate,
            source_lang=source_language,
            target_langs=target_language_codes
        )

        # Check if translation failed completely after all fallbacks
        if translations is None:
            logger.error(f"All translation attempts failed for message {message.id}.")
            # Optionally notify user or channel, but avoid spamming if failures are frequent
            # await message.channel.send(f"Sorry {message.author.mention}, translation failed after trying all available services. Please check logs or contact admin.", delete_after=20)
            return # Stop processing if translation failed

        logger.info(f"Successfully received translation results (potentially via fallback) for message {message.id}.")

        # --- Distribute Translation Results ---
        # Pass the potentially empty dict {} if translation resulted in missing languages
        await self.distribute_translations(message, source_language, translations, writable_target_channels)

    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edit events if message is cached (Fallback)."""
        logger.debug(f"on_message_edit triggered for message ID {after.id} (might be ignored).")
        if after.author.bot: return
        source_channel_id = after.channel.id
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only' or source_channel_id not in self.channel_language_map: return
        if before and before.content is not None and before.content == after.content:
             logger.debug(f"[on_message_edit] Content of message {after.id} did not change. Skipping.")
             return
        logger.info(f"[on_message_edit] Processing edit for message {after.id} (potentially redundant, check raw handler logs).")


    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        """Handles raw message edit events directly from the gateway (Primary Handler)."""
        logger.debug(f"on_raw_message_edit triggered for message ID {payload.message_id}")

        if not payload.channel_id or not payload.message_id:
            logger.debug("Raw edit ignored: Missing channel_id or message_id.")
            return

        # Check Channel Configuration
        source_channel_id = payload.channel_id
        source_channel_type = self.channel_type_map.get(source_channel_id)
        if source_channel_type == 'write_only': return
        if source_channel_id not in self.channel_language_map: return

        # Extract Edited Content
        new_content = payload.data.get('content', None)
        if new_content is None: return # Only handle content edits

        # Check if Author is Bot
        author_data = payload.data.get('author', {})
        if author_data.get('bot', False): return

        # --- Check Database for Tracking ---
        if not self.db: logger.error("Database not available in on_raw_message_edit."); return

        tracked_translations = []
        try:
            query = "SELECT target_language, translated_message_id, target_channel_id FROM message_map WHERE original_message_id = ?"
            params = (payload.message_id,)
            async with self.db.execute(query, params) as cursor:
                tracked_translations = await cursor.fetchall()
        except Exception as e:
            logger.error(f"Error querying database for message map of {payload.message_id}: {e}", exc_info=True)
            return

        if not tracked_translations: return # Not tracking this message
        # --- End Database Check ---

        # --- Proceed with Re-translation and Update ---
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language: logger.warning(f"Internal error: Could not find language for tracked message {payload.message_id}."); return

        logger.info(f"Detected edit via raw event in tracked message {payload.message_id} in channel {source_channel_id} ({source_language}).")

        target_language_codes = [row[0] for row in tracked_translations]
        if not target_language_codes: logger.warning(f"No target languages found in DB map for edited message {payload.message_id}."); return

        log_content = new_content[:100] + ('...' if len(new_content) > 100 else '')
        logger.info(f"Re-translating edited content to languages: {target_language_codes}")

        # --- Perform Re-translation using Fallback System ---
        new_translations = await self.translate_with_fallback(
            text=new_content,
            source_lang=source_language,
            target_langs=target_language_codes
        )

        if new_translations is None:
            logger.error(f"Failed to re-translate content for edited message {payload.message_id} after trying all fallbacks.")
            # Optionally notify? Probably not for edits to avoid noise.
            return # Stop if re-translation failed

        logger.info(f"Successfully received updated translation results (potentially via fallback) for edited message {payload.message_id}.")

        # Update the messages using the fetched tracking info
        await self.update_translated_messages(payload.message_id, new_translations, tracked_translations)


    # Note: translate_text_with_openai is kept as it was used internally by the old on_message/on_edit logic.
    # The new logic uses _translate_with_service internally. We can potentially remove translate_text_with_openai
    # later if no other part of the code relies on it directly. For now, it remains but is unused by the main flow.
    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """[DEPRECATED by translate_with_fallback] Translates text using the (legacy) single configured OpenAI client."""
        logger.warning("translate_text_with_openai called directly. This method is deprecated in favor of translate_with_fallback.")
        # Attempt to use the primary client if available, mimicking old behavior somewhat
        if 'primary' in self.translation_clients:
             logger.debug("Forwarding deprecated call to _translate_with_service using 'primary' client.")
             return await self._translate_with_service('primary', text, source_lang, target_langs)
        else:
             logger.error("Deprecated translate_text_with_openai called, but no 'primary' client is initialized.")
             return None

        # --- The original logic below is now effectively superseded by _translate_with_service ---
        # if not self.openai_client: logger.error("OpenAI client not initialized."); return None
        # model_name = self.settings.get('translation_model', 'google/gemma-3-27b-it')
        # system_prompt = f"""You are an expert multilingual translator. Translate the user's text from {source_lang} into the following languages: {', '.join(target_langs)}.
# Respond ONLY with a valid JSON object containing the translations. The JSON object should have language codes (exactly as provided: {', '.join(target_langs)}) as keys and the corresponding translated text as string values.
# Example format for targets {target_langs}: {{ "{target_langs[0]}": "translation for {target_langs[0]}", "{target_langs[1]}": "translation for {target_langs[1]}" }}
# Ensure the output is nothing but the JSON object. Do not include any explanations, markdown formatting around the JSON, or introductory text. Preserve original formatting like markdown (e.g., bold, italics) within the translated strings where appropriate, but prioritize accurate translation. If the input text is empty or contains only whitespace, return an empty JSON object {{}}.
# Target languages: {target_langs}"""
#         user_prompt = text if text else ""
#         content_str = None
#         try:
#             response = await self.openai_client.chat.completions.create(
#                 model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
#                 response_format={"type": "json_object"}, temperature=0.5, max_tokens=1500 )
#             content_str = response.choices[0].message.content
#             if not content_str: logger.error("API response successful, but content is empty."); return None
#             # Clean Markdown
#             cleaned_content_str = content_str.strip()
#             if cleaned_content_str.startswith("```json"): cleaned_content_str = cleaned_content_str[len("```json"):].strip()
#             if cleaned_content_str.startswith("```"): cleaned_content_str = cleaned_content_str[len("```"):].strip()
#             if cleaned_content_str.endswith("```"): cleaned_content_str = cleaned_content_str[:-len("```")].strip()
#             if not cleaned_content_str: logger.error("Content is empty after cleaning Markdown symbols."); logger.debug(f"Original content_str: {content_str}"); return None
#             translations = json.loads(cleaned_content_str)
#             if not isinstance(translations, dict): logger.error(f"Parsed content is not a valid JSON object: {translations}"); return None
#             # Fill missing
#             missing_langs = [lang for lang in target_langs if lang not in translations]
#             if missing_langs:
#                 logger.warning(f"API response missing translations for some languages: {missing_langs}")
#                 for lang in missing_langs: translations[lang] = f"[{lang.upper()} translation missing]"
#             # Filter extra
#             filtered_translations = {k: v for k, v in translations.items() if k in target_langs}
#             return filtered_translations
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse API response JSON: {e}")
#             original_content_to_log = content_str if content_str is not None else "[Could not get raw response content]"
#             logger.error(f"Raw response content: {original_content_to_log}")
#             return None
#         except (AttributeError, IndexError, TypeError) as e:
#              logger.error(f"Error processing API response structure: {e}")
#              response_to_log = response if 'response' in locals() else "[Could not get response object]"
#              logger.debug(f"Full API response object: {response_to_log}")
#              return None
#         except Exception as e:
#             logger.error(f"Unexpected error during translation using OpenAI library: {e}", exc_info=True)
#             return None


    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, writable_target_map: dict):
        """
        Distribute translations and store mappings in the database using the new format.
        """
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []
        lang_codes_in_order = list(translations.keys()) if translations else list(writable_target_map.keys())

        # --- Modified: Inner function uses new format ---
        async def _send_translation(target_channel_id, lang_code, translated_text):
            sent_msg_id = None
            try:
                target_channel = self.get_channel(target_channel_id)
                channel_type = self.channel_type_map.get(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel) and channel_type in ['standard', 'write_only']:
                    # Create embed with translated text (or None if no text)
                    embed_to_send = discord.Embed(
                        description=translated_text if translated_text and translated_text.strip() else None,
                        color=discord.Color.blue()
                    )
                    # Set author field
                    embed_to_send.set_author(
                        name=f"{original_author.display_name} (from #{original_channel.name})",
                        icon_url=original_author.display_avatar.url if original_author.display_avatar else None
                    )
                    # Add single compact hyperlink with source language
                    embed_to_send.add_field(
                        name="", # Keep name empty for cleaner look
                        value=f"[Original Message ({source_language.upper()})]({original_message.jump_url})",
                        inline=False
                    )

                    # Handle image attachment directly in embed if present
                    if original_message.attachments:
                        first_att = original_message.attachments[0]
                        filename_lower = first_att.filename.lower()
                        # Check for common image extensions
                        if any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            embed_to_send.set_image(url=first_att.proxy_url) # Use proxy_url

                        # If multiple attachments, add compact reference link
                        if len(original_message.attachments) > 1:
                            att_count = len(original_message.attachments)
                            # Add a field linking back to the original message for other attachments
                            embed_to_send.add_field(
                                name="", # Keep name empty
                                value=f"[+{att_count-1} more attachment{'s' if att_count > 2 else ''}]({original_message.jump_url})",
                                inline=False
                            )
                    # --- End Format Changes ---

                    sent_msg = await target_channel.send(embed=embed_to_send)
                    sent_msg_id = sent_msg.id
                    logger.info(f"Sent translation ({lang_code}) to {channel_type} channel '{target_channel.name}' ({target_channel_id}) - Msg ID: {sent_msg_id}")
                elif not target_channel or not isinstance(target_channel, discord.TextChannel): logger.warning(f"Could not find target channel ID {target_channel_id}.")
                elif channel_type == 'read_only': logger.warning(f"Attempted send to read_only channel {target_channel_id}. Skipping.")
            except discord.errors.Forbidden: logger.error(f"Permission denied: Cannot send message to channel ID {target_channel_id}.")
            except discord.errors.HTTPException as e: logger.error(f"HTTP error sending message to channel ID {target_channel_id} (Status: {e.status}): {e.text}")
            except Exception as e: logger.error(f"Unexpected error distributing translation to channel ID {target_channel_id}: {e}", exc_info=True)
            return {'msg_id': sent_msg_id, 'channel_id': target_channel_id} if sent_msg_id else None
        # --- End Modified Inner Function ---

        # Create send tasks (logic unchanged)
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
            # Store mapping in Database (logic unchanged)
            if not self.db: logger.error("Database not available, cannot store message map."); return

            insert_data_list = []
            for i, result_detail in enumerate(results):
                if result_detail and result_detail.get('msg_id'):
                    lang_code = lang_codes_in_order[i]
                    translated_msg_id = result_detail['msg_id']
                    target_channel_id = result_detail['channel_id']
                    insert_data_list.append((original_message.id, lang_code, translated_msg_id, target_channel_id))

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

    def calculate_effective_length(self, text: str) -> int:
        """
        Calculates the 'effective' length of the text, giving higher weight to CJK characters.
        This helps estimate potential token usage more accurately for separation logic.
        """
        effective_length = 0
        for char in text:
            # Check if character is CJK (Unicode ranges)
            # CJK Unified Ideographs: U+4E00 to U+9FFF
            # Hangul Syllables: U+AC00 to U+D7AF
            # Hiragana: U+3040 to U+309F
            # Katakana: U+30A0 to U+30FF
            # CJK Symbols and Punctuation: U+3000 to U+303F
            # Halfwidth and Fullwidth Forms: U+FF00 to U+FFEF (includes fullwidth Latin chars)
            if (
                '\u4e00' <= char <= '\u9fff' or
                '\uac00' <= char <= '\ud7af' or
                '\u3040' <= char <= '\u309f' or
                '\u30a0' <= char <= '\u30ff' or
                '\u3000' <= char <= '\u303f' or
                '\uff00' <= char <= '\uffef'
            ):
                effective_length += 2 # Assign double weight
            else:
                effective_length += 1 # Assign single weight
        return effective_length

    async def update_translated_messages(self, original_message_id: int, new_translations: dict, tracked_translations: list):
        """
        Finds and edits translated messages, preserving the new format.
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
                    # --- Modified: Only update description ---
                    # Create a new embed by copying, ensuring other fields (author, image, links) are preserved
                    new_embed = original_embed.copy()
                    # Update only the description with the new translation (or None if no text)
                    new_embed.description = new_text if new_text and new_text.strip() else None
                    # --- End Modification ---
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

    # --- New Translation Service Methods ---

    def _init_openai_client(self, service_id, config):
        """Initializes an OpenAI-compatible client based on config."""
        try:
            provider = config.get('provider', 'unknown')
            base_url = config.get('base_url')
            api_key = config.get('api_key')

            if not base_url or not api_key:
                logger.error(f"Missing base_url or api_key for {provider} service ({service_id}). Cannot initialize.")
                return False

            # Handle Azure-specific parameters if needed (though AsyncOpenAI might handle them)
            # api_version = config.get('api_version')
            # deployment_name = config.get('deployment_name') # May need specific handling if AsyncOpenAI doesn't auto-detect

            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                # Potentially add http_client or other options if needed later
            )

            self.translation_clients[service_id] = {
                'client': client,
                'provider': provider,
                'model': config.get('model'),
                'config': config # Store original config for reference
            }

            logger.info(f"Initialized {provider} client for service ID: {service_id} (Model: {config.get('model')})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {service_id} client ({config.get('provider', 'unknown')}): {e}", exc_info=True)
            return False

    async def translate_with_fallback(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """Attempts translation using primary, then remote fallbacks, then Ollama."""
        if not text or not target_langs:
            logger.warning("translate_with_fallback called with empty text or target_langs.")
            return {} # Return empty dict for consistency

        # --- Language Separation Logic ---
        effective_length = self.calculate_effective_length(text)
        threshold = self.settings.get('language_separation_threshold', 800) # Get threshold from settings
        logger.debug(f"Effective length: {effective_length}, Threshold: {threshold}")

        if effective_length > threshold:
            logger.info(f"Text effective length ({effective_length}) exceeds threshold ({threshold}). Processing languages separately.")
            combined_translations = {}
            successful_langs = []
            failed_langs = []

            for lang in target_langs:
                logger.info(f"Separated processing: Translating to '{lang}'...")
                single_lang_target = [lang]
                single_translation = None

                # Try primary service for this single language
                if 'primary' in self.translation_clients:
                    single_translation = await self._translate_with_service('primary', text, source_lang, single_lang_target)
                    if single_translation is None: logger.warning(f"Separated processing: Primary service failed for '{lang}'.")

                # Try remote fallbacks if primary failed
                if single_translation is None and self.enable_remote_fallback and 'remote_fallbacks' in self.translation_services:
                    num_fallbacks = len(self.translation_services['remote_fallbacks'])
                    for i in range(num_fallbacks):
                        service_id = f"fallback_{i}"
                        if service_id in self.translation_clients:
                            provider = self.translation_clients[service_id].get('provider', 'unknown')
                            logger.info(f"Separated processing: Trying remote fallback {i+1}/{num_fallbacks} ({provider}) for '{lang}'...")
                            single_translation = await self._translate_with_service(service_id, text, source_lang, single_lang_target)
                            if single_translation is not None:
                                logger.info(f"Separated processing: Remote fallback {service_id} ({provider}) succeeded for '{lang}'.")
                                break # Stop trying fallbacks for this language if one succeeds
                            else:
                                logger.warning(f"Separated processing: Remote fallback {service_id} ({provider}) failed for '{lang}'.")

                # Try Ollama if remote fallbacks failed (or weren't enabled/configured)
                if single_translation is None and self.enable_ollama_fallback:
                    logger.info(f"Separated processing: Trying Ollama fallback for '{lang}'...")
                    single_translation = await self.translate_text_with_ollama(text, source_lang, single_lang_target)
                    if single_translation is not None:
                        logger.info(f"Separated processing: Ollama fallback succeeded for '{lang}'.")
                    else:
                        logger.warning(f"Separated processing: Ollama fallback failed for '{lang}'.")

                # Combine results
                if single_translation is not None and lang in single_translation:
                    combined_translations[lang] = single_translation[lang]
                    successful_langs.append(lang)
                else:
                    logger.error(f"Separated processing: All attempts failed for language '{lang}'.")
                    failed_langs.append(lang)
                    combined_translations[lang] = f"[{lang.upper()} translation failed]" # Add placeholder for failed ones

            logger.info(f"Finished separated language processing. Success: {successful_langs}, Failed: {failed_langs}")
            return combined_translations # Return combined results, including placeholders for failures

        # --- Original Logic (if below threshold) ---
        else:
            logger.info(f"Text effective length ({effective_length}) is within threshold ({threshold}). Processing all languages together.")
            # 1. Try Primary Service
            if 'primary' in self.translation_clients:
                logger.info(f"Attempting translation with primary service (ID: primary)...")
                result = await self._translate_with_service('primary', text, source_lang, target_langs) # Correctly indented now
                if result is not None: # Check for None explicitly, as {} is a valid (empty) result
                    logger.info("Primary translation service succeeded.")
                    return result # Return immediately if primary succeeds
                else:
                    logger.warning("Primary translation service failed.")
            else: # Correctly indented else corresponding to 'if primary in ...'
                logger.error("Primary translation service client not initialized. Cannot attempt primary translation.")
                # Do not return here, proceed to fallbacks

            # 2. Try Remote Fallback Services (if enabled and primary failed or wasn't available)
            if self.enable_remote_fallback and 'remote_fallbacks' in self.translation_services:
                num_fallbacks = len(self.translation_services['remote_fallbacks']) # Correctly indented
                logger.info(f"Attempting translation with {num_fallbacks} remote fallback service(s)...") # Correctly indented
                for i in range(num_fallbacks): # Correctly indented
                    service_id = f"fallback_{i}" # Correctly indented
                    if service_id in self.translation_clients: # Correctly indented
                        provider = self.translation_clients[service_id].get('provider', 'unknown') # Correctly indented
                        logger.info(f"Attempting translation with remote fallback service {i+1}/{num_fallbacks} (ID: {service_id}, Provider: {provider})...") # Correctly indented
                        result = await self._translate_with_service(service_id, text, source_lang, target_langs) # Correctly indented
                        if result is not None: # Correctly indented
                            logger.info(f"Remote fallback service {service_id} ({provider}) succeeded.") # Correctly indented
                            return result # Correctly indented
                        else: # Correctly indented
                            logger.warning(f"Remote fallback service {service_id} ({provider}) failed.") # Correctly indented
                    else: # Correctly indented
                        logger.warning(f"Remote fallback service {service_id} was configured but not initialized. Skipping.") # Correctly indented

            # 3. Try Ollama Local Fallback (if enabled and previous steps failed)
            if self.enable_ollama_fallback:
                logger.info("Attempting translation with Ollama local fallback...") # Correctly indented
                result = await self.translate_text_with_ollama(text, source_lang, target_langs) # Correctly indented
                if result is not None: # Correctly indented
                    logger.info("Ollama local fallback succeeded.") # Correctly indented
                    return result # Correctly indented
                else: # Correctly indented
                    logger.warning("Ollama local fallback failed.") # Correctly indented

            # All services failed if we reach here
            logger.error("All configured translation services (primary, remote fallbacks, Ollama) failed.")
        return None # Indicate total failure

    async def _translate_with_service(self, service_id: str, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """Uses a specific initialized OpenAI-compatible service for translation."""
        if service_id not in self.translation_clients:
            logger.error(f"Translation service '{service_id}' not initialized or found.")
            return None

        service_info = self.translation_clients[service_id]
        client = service_info['client']
        model = service_info['model']
        provider = service_info['provider']

        if not client or not model:
            logger.error(f"Client or model missing for service '{service_id}' ({provider}).")
            return None

        logger.debug(f"Using service '{service_id}' ({provider}) with model '{model}'")

        # --- Dynamically construct the example format for the prompt ---
        example_pairs = []
        if len(target_langs) >= 1:
            example_pairs.append(f'"{target_langs[0]}": "translation for {target_langs[0]}"')
        if len(target_langs) >= 2:
            example_pairs.append(f'"{target_langs[1]}": "translation for {target_langs[1]}"')
        # Add more examples if desired, or handle the case of 0 target_langs if necessary
        example_format_str = f"{{ {', '.join(example_pairs)} }}" if example_pairs else "{}"
        # --- End dynamic example construction ---

        # Construct the prompt using the dynamic example
        system_prompt = f"""You are an expert multilingual translator. Translate the user's text from {source_lang} into the following languages: {', '.join(target_langs)}.
Respond ONLY with a valid JSON object containing the translations. The JSON object should have language codes (exactly as provided: {', '.join(target_langs)}) as keys and the corresponding translated text as string values.
Example format for targets {target_langs}: {example_format_str}
Ensure the output is nothing but the JSON object. Do not include any explanations, markdown formatting around the JSON, or introductory text. Preserve original formatting like markdown (e.g., bold, italics) within the translated strings where appropriate, but prioritize accurate translation. If the input text is empty or contains only whitespace, return an empty JSON object {{}}.
Target languages: {target_langs}"""
        user_prompt = text if text else ""
        content_str = None

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=6000 # Consider making this configurable per service later if needed
            )
            content_str = response.choices[0].message.content
            if not content_str:
                logger.error(f"API response from service '{service_id}' ({provider}) successful, but content is empty.")
                return None # Treat empty content as failure for this service

            # Clean potential Markdown formatting around JSON
            cleaned_content_str = content_str.strip()
            if cleaned_content_str.startswith("```json"): cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            if cleaned_content_str.startswith("```"): cleaned_content_str = cleaned_content_str[len("```"):].strip()
            if cleaned_content_str.endswith("```"): cleaned_content_str = cleaned_content_str[:-len("```")].strip()

            if not cleaned_content_str:
                logger.error(f"Content from service '{service_id}' ({provider}) is empty after cleaning Markdown symbols.")
                logger.debug(f"Original content_str from {service_id}: {content_str}")
                return None

            # Parse JSON with fallback for truncation
            translations = None
            try:
                translations = json.loads(cleaned_content_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed for service '{service_id}' ({provider}): {e}. Possible truncation. Attempting recovery.")
                logger.debug(f"Raw content before recovery attempt: {cleaned_content_str}")
                # Attempt to find the last valid JSON object part
                last_brace_index = cleaned_content_str.rfind('}')
                if last_brace_index != -1:
                    truncated_json_str = cleaned_content_str[:last_brace_index + 1]
                    # Add heuristics: Check if it looks somewhat like a JSON object ending
                    if '"' in truncated_json_str and ':' in truncated_json_str:
                        logger.info(f"Attempting to parse potentially truncated JSON: {truncated_json_str}")
                        try:
                            translations = json.loads(truncated_json_str)
                            logger.info(f"Successfully parsed truncated JSON from service '{service_id}' ({provider}).")
                        except json.JSONDecodeError as e_recovery:
                            logger.error(f"Failed to parse even the truncated JSON from service '{service_id}' ({provider}): {e_recovery}")
                            logger.debug(f"Truncated string attempted: {truncated_json_str}")
                            # Optional: Add regex fallback here if needed as a last resort
                            # json_match = re.search(r'\{.*\}', cleaned_content_str, re.DOTALL) ... etc.
                    else:
                         logger.error(f"Truncated string for service '{service_id}' ({provider}) does not appear to be a valid JSON fragment ending.")
                         logger.debug(f"Truncated string attempted: {truncated_json_str}")

                else:
                    logger.error(f"Could not find a closing brace '}}' in the response from service '{service_id}' ({provider}) to attempt recovery.")

                if translations is None: # If recovery failed
                     return None # Give up for this service

            # --- End JSON Parsing Fallback ---

            if not isinstance(translations, dict):
                logger.error(f"Parsed content from service '{service_id}' ({provider}) is not a valid JSON object (even after potential recovery): {translations}")
                return None

            # Validate and filter response (ensure all target langs are present, filter extras)
            final_translations = {}
            missing_langs = []
            for lang in target_langs:
                if lang in translations and isinstance(translations[lang], str):
                    final_translations[lang] = translations[lang]
                else:
                    missing_langs.append(lang)
                    final_translations[lang] = f"[{lang.upper()} translation missing or invalid]" # Placeholder

            if missing_langs:
                logger.warning(f"Service '{service_id}' ({provider}) response missing or invalid translations for: {missing_langs}")

            # Log if extra languages were returned (optional)
            extra_langs = [k for k in translations if k not in target_langs]
            if extra_langs:
                logger.debug(f"Service '{service_id}' ({provider}) returned extra languages not requested: {extra_langs}")

            return final_translations # Return the validated/filtered dictionary

        # except json.JSONDecodeError as e: # This is now handled by the try/except block above
        #     logger.error(f"Failed to parse JSON response from service '{service_id}' ({provider}): {e}")
        #     original_content_to_log = content_str if content_str is not None else "[Could not get raw response content]"
        #     logger.error(f"Raw response content from {service_id}: {original_content_to_log}")
        #     return None
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.AuthenticationError, openai.APITimeoutError, openai.BadRequestError) as e: # Added Timeout/BadRequest
             # Catch specific OpenAI errors (or compatible errors)
             logger.error(f"API error with service '{service_id}' ({provider}): {type(e).__name__} - Status: {getattr(e, 'status_code', 'N/A')} - {e}")
             return None
        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"Error processing API response structure from service '{service_id}' ({provider}): {e}")
             response_to_log = response if 'response' in locals() else "[Could not get response object]"
             logger.debug(f"Full API response object from {service_id}: {response_to_log}")
             return None
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during translation with service '{service_id}' ({provider}): {e}", exc_info=True)
            return None

    async def translate_text_with_ollama(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """Translates text using a local Ollama model via its API."""
        if not self.ollama_config:
            logger.error("Ollama configuration is missing. Cannot use Ollama fallback.")
            return None

        base_url = self.ollama_config.get('base_url')
        model = self.ollama_config.get('model')
        timeout_seconds = self.ollama_config.get('timeout', 30)

        if not base_url or not model:
            logger.error("Ollama base_url or model is not configured.")
            return None

        # Construct the prompt for Ollama (similar structure, might need tuning per model)
        system_prompt = f"""You are an expert multilingual translator. Translate the text from {source_lang} into the following languages: {', '.join(target_langs)}.
Respond ONLY with a valid JSON object containing the translations. Format: {{"lang_code": "translation"}}
Example for targets {target_langs}: {{ "{target_langs[0]}": "translation for {target_langs[0]}", "{target_langs[1]}": "translation for {target_langs[1]}" }}
Ensure the output is nothing but the JSON object. Do not include explanations or introductory text."""

        full_prompt = f"{system_prompt}\n\nText to translate:\n{text}"
        api_endpoint = f"{base_url.rstrip('/')}/api/generate" # Use /api/generate for non-streaming

        logger.debug(f"Sending request to Ollama: Endpoint={api_endpoint}, Model={model}")

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
                async with session.post(
                    api_endpoint,
                    json={"model": model, "prompt": full_prompt, "stream": False, "format": "json"}, # Request JSON format directly if model supports it
                ) as response:
                    response_text = await response.text() # Get text first for logging
                    if response.status != 200:
                        logger.error(f"Ollama API error: Status {response.status}, Response: {response_text}")
                        return None

                    # Attempt to parse the JSON response
                    try:
                        result = json.loads(response_text)
                        # Ollama's non-streaming JSON response is typically in result['response'] as a stringified JSON
                        response_content_str = result.get('response')
                        if not response_content_str:
                             logger.error("Ollama response key 'response' is missing or empty.")
                             logger.debug(f"Full Ollama JSON result: {result}")
                             return None

                        # Parse the inner JSON string
                        translations = json.loads(response_content_str)
                        if not isinstance(translations, dict):
                            logger.error(f"Parsed inner content from Ollama is not a dict: {translations}")
                            return None

                        # Validate and filter response (similar to _translate_with_service)
                        final_translations = {}
                        missing_langs = []
                        for lang in target_langs:
                            if lang in translations and isinstance(translations[lang], str):
                                final_translations[lang] = translations[lang]
                            else:
                                missing_langs.append(lang)
                                final_translations[lang] = f"[{lang.upper()} translation missing or invalid]"

                        if missing_langs:
                            logger.warning(f"Ollama response missing or invalid translations for: {missing_langs}")

                        return final_translations

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from Ollama response: {e}")
                        # Try regex as a fallback if direct JSON parsing fails (e.g., model didn't respect format="json")
                        logger.info("Attempting regex fallback for Ollama JSON extraction...")
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            try:
                                translations = json.loads(json_str)
                                if isinstance(translations, dict):
                                     # Validate and filter again
                                    final_translations = {}
                                    missing_langs = []
                                    for lang in target_langs:
                                        if lang in translations and isinstance(translations[lang], str):
                                            final_translations[lang] = translations[lang]
                                        else:
                                            missing_langs.append(lang)
                                            final_translations[lang] = f"[{lang.upper()} translation missing or invalid]"
                                    if missing_langs: logger.warning(f"Ollama (regex fallback) missing/invalid translations for: {missing_langs}")
                                    logger.info("Ollama regex fallback JSON extraction successful.")
                                    return final_translations
                                else:
                                     logger.error("Ollama regex fallback extracted content is not a dict.")
                            except json.JSONDecodeError as e_regex:
                                logger.error(f"Failed to parse JSON even from regex fallback: {e_regex}")
                                logger.debug(f"Ollama raw response (regex failed): {response_text}")
                                return None
                        else:
                            logger.error("Could not find any JSON object in Ollama response via regex.")
                            logger.debug(f"Ollama raw response (regex failed): {response_text}")
                            return None
                    except Exception as e_inner:
                         logger.error(f"Error processing Ollama response content: {e_inner}")
                         logger.debug(f"Ollama raw response: {response_text}")
                         return None

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Ollama connection error: Could not connect to {api_endpoint}. Is Ollama running? Error: {e}")
            return None
        except asyncio.TimeoutError:
             logger.error(f"Ollama request timed out after {timeout_seconds} seconds.")
             return None
        except aiohttp.ClientError as e:
            logger.error(f"Ollama client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Ollama translation: {e}", exc_info=True)
            return None


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

    # Validate config (Updated for new structure)
    api_keys = temp_config.get('api_keys', {})
    channels_config = temp_config.get('channels', {})
    translation_services_config = temp_config.get('translation_services', {})
    discord_token = os.getenv('DISCORD_BOT_TOKEN') or api_keys.get('discord_token')

    config_valid = True
    # Check Discord Token
    if not discord_token or discord_token == 'YOUR_DISCORD_BOT_TOKEN':
        logger.critical("Error: Valid Discord Bot Token not configured in api_keys.")
        config_valid = False

    # Check Channels Config
    if not isinstance(channels_config, dict) or not channels_config:
        logger.critical("Error: 'channels' section missing, invalid, or empty in config.yaml.")
        config_valid = False
    elif not any(langs for type_data in channels_config.values() if isinstance(type_data, dict) for langs in type_data.values()):
        logger.critical("Error: No channels defined under standard, read_only, or write_only in config.yaml.")
        config_valid = False

    # Check Translation Services Config (Basic checks)
    if not isinstance(translation_services_config, dict) or not translation_services_config:
         logger.critical("Error: 'translation_services' section missing or invalid in config.yaml.")
         config_valid = False
    else:
        primary_config = translation_services_config.get('primary')
        if not isinstance(primary_config, dict) or not primary_config.get('api_key') or primary_config['api_key'] == 'YOUR_OPENROUTER_API_KEY': # Example check for primary
             logger.critical("Error: Primary translation service ('primary') is not configured correctly (missing or default api_key).")
             # We don't strictly make this False yet, as fallback might be intended, but it's a critical warning.
             # config_valid = False # Uncomment if primary MUST be valid

        # Add more checks here if needed (e.g., for fallback keys if enabled)

    if not config_valid:
        logger.critical("Configuration validation failed. Please check config.yaml and restart.")
        exit(1)

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
