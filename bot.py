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
from collections import defaultdict # To store message mappings

# --- Configuration ---
CONFIG_FILE = 'config.yaml'
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('TranslationBot')

# OpenAI compatible base URL for OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Core Bot Class ---
class TranslationBot(discord.Client):
    """
    Translation bot class inheriting from discord.Client.
    Includes message edit tracking.
    """
    def __init__(self, *, intents: discord.Intents, config_path: str):
        """
        Initialize the Bot.

        Args:
            intents (discord.Intents): Required permission intents for the Bot.
            config_path (str): Path to the configuration file.
        """
        super().__init__(intents=intents)
        self.config_path = config_path
        self.channel_language_map = {} # Structure: {channel ID (int): language code (str)}
        self.language_channel_map = {} # Structure: {language code (str): [channel ID (int), ...]}
        self.api_keys = {}             # Structure: {'discord_token': '...', 'openrouter_key': '...'}
        self.settings = {              # Default settings
            'translation_model': 'google/gemini-flash',
            'max_retries': 2,
            'retry_delay': 3
        }
        self.openai_client: AsyncOpenAI | None = None # OpenAI client
        # --- Added: In-memory storage for message mappings ---
        # Structure: {original_message_id: {target_lang_code: translated_message_id, ...}}
        self.message_map = defaultdict(dict)
        # Limit the size to prevent memory issues (optional)
        self.message_map_max_size = 10000

    async def setup_hook(self) -> None:
        """
        Asynchronous setup hook called before the Bot starts.
        """
        self.load_config()
        logger.info(f'Configuration loaded. Listening on {len(self.channel_language_map)} channels.')
        logger.info(f"Using model '{self.settings.get('translation_model')}' for translation.")
        # Initialize OpenAI client
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
        """Clean up resources when the Bot is closing."""
        await super().close()
        logger.info("Bot is shutting down.")

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
                if 'channels' in config_data and isinstance(config_data['channels'], dict):
                    self.channel_language_map = {}
                    self.language_channel_map = {}
                    for lang_code, channel_ids in config_data['channels'].items():
                        if not isinstance(channel_ids, list): channel_ids = [channel_ids]
                        if lang_code not in self.language_channel_map: self.language_channel_map[lang_code] = []
                        for channel_id in channel_ids:
                            try:
                                ch_id_int = int(channel_id)
                                if ch_id_int in self.channel_language_map: logger.warning(f"Channel ID {ch_id_int} is configured for both '{self.channel_language_map[ch_id_int]}' and '{lang_code}'. Using the latter ('{lang_code}').")
                                self.channel_language_map[ch_id_int] = lang_code
                                self.language_channel_map[lang_code].append(ch_id_int)
                            except ValueError: logger.warning(f"Invalid channel ID '{channel_id}' found in config file (language: {lang_code}). Skipping.")
                    logger.info(f"Successfully loaded {len(self.channel_language_map)} channel configurations.")
                else: logger.warning(f"Missing or invalid 'channels' section in configuration file '{self.config_path}'.")
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
            'channels': {'en': [123456789012345678], 'zh-TW': [987654321098765432], 'ja': [555555555555555555]},
            'api_keys': {'discord_token': 'YOUR_DISCORD_BOT_TOKEN', 'openrouter_key': 'YOUR_OPENROUTER_API_KEY'},
            'settings': {'translation_model': 'google/gemini-flash', 'max_retries': 2, 'retry_delay': 3}
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f: yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            logger.info(f"Example configuration file '{self.config_path}' created. Please fill in your information.")
        except Exception as e: logger.error(f"Error creating example configuration file '{self.config_path}': {e}")

    async def on_ready(self):
        """Called when the Bot is ready."""
        logger.info(f'Logged in successfully as {self.user} (ID: {self.user.id})')
        logger.info('Bot is ready.')

    async def on_message(self, message: discord.Message):
        """Handle new message events."""
        # Ignore messages from self or other bots
        if message.author == self.user or message.author.bot: return
        # Ignore messages not from configured channels
        source_channel_id = message.channel.id
        if source_channel_id not in self.channel_language_map: return
        # Ignore messages with no content and no attachments
        if not message.content and not message.attachments:
            logger.debug(f"Message from channel {message.channel.name} has empty content and no attachments. Ignoring.")
            return

        # Get source language from configuration
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language:
            logger.warning(f"Internal error: Could not find language setting for channel ID {source_channel_id} in channel_language_map.")
            return

        # Determine target languages and channels
        target_languages_map = {}
        target_language_codes = []
        for lang_code, channel_ids in self.language_channel_map.items():
            if lang_code != source_language:
                target_languages_map[lang_code] = channel_ids
                target_language_codes.append(lang_code)

        if not target_languages_map:
            logger.info(f"No target language channels configured for messages from channel {message.channel.name} ({source_language}).")
            return

        text_to_translate = message.content
        # If only attachments, sync links without translation
        if not text_to_translate and message.attachments:
             logger.info(f"Message {message.id} contains only attachments. Syncing attachment links only.")
             await self.distribute_translations(message, source_language, {}, target_languages_map)
             return

        logger.info(f"Detected message from channel '{message.channel.name}' ({source_channel_id}, Language: {source_language}), Author: {message.author.display_name}")
        log_content = text_to_translate[:100] + ('...' if len(text_to_translate) > 100 else '')
        logger.info(f"Preparing to translate content to languages: {target_language_codes}")
        logger.debug(f"Content preview for translation: '{log_content}'")

        # --- Perform Translation ---
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized. Cannot perform translation.")
                raise RuntimeError("OpenAI client not initialized")

            translations = await self.translate_text_with_openai(
                text=text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )

            if translations is None:
                await message.channel.send(f"Sorry {message.author.mention}, there was a problem during translation. Could not complete sync. Please check logs for details.", delete_after=15)
                return

            logger.info(f"Successfully received translation results from API.")
            logger.debug(f"Translation results: {translations}")

        # Handle specific OpenAI errors
        except openai.AuthenticationError:
             logger.error("OpenAI API Authentication Failed: Invalid API Key or insufficient permissions.")
             await message.channel.send(f"Sorry {message.author.mention}, translation service authentication failed. Please contact the administrator.", delete_after=15)
             return
        except openai.RateLimitError:
            logger.warning("OpenAI API Rate Limit Exceeded: Too many requests.")
            await message.channel.send(f"Sorry {message.author.mention}, translation requests are too frequent. Please try again later.", delete_after=15)
            return
        except openai.APIConnectionError as e:
            logger.error(f"Could not connect to OpenAI API: {e}")
            await message.channel.send(f"Sorry {message.author.mention}, could not connect to the translation service.", delete_after=15)
            return
        except openai.APIError as e:
             logger.error(f"OpenAI API returned an error: {e}")
             await message.channel.send(f"Sorry {message.author.mention}, the translation service encountered an error.", delete_after=15)
             return
        # Handle other unexpected errors during translation
        except Exception as e:
            logger.error(f"Unexpected error during translation process: {e}", exc_info=True)
            await message.channel.send(f"Sorry {message.author.mention}, an internal error occurred during translation.", delete_after=15)
            return

        # --- Distribute Translation Results ---
        # The distribute_translations function will now return the mapping
        await self.distribute_translations(message, source_language, translations, target_languages_map)

    # --- Added: Handle message edits ---
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edit events."""
        # Ignore edits from bots or if content hasn't changed
        if after.author.bot or before.content == after.content:
            return

        # Ignore edits in channels not configured for translation
        source_channel_id = after.channel.id
        if source_channel_id not in self.channel_language_map:
            return

        # Ignore edits to messages we haven't tracked (not in message_map)
        if after.id not in self.message_map:
            logger.debug(f"Edited message {after.id} not found in tracking map. Ignoring edit.")
            return

        # Get original source language
        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language: # Should not happen if message is in map
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

        # --- Perform Re-Translation ---
        try:
            if not self.openai_client:
                logger.error("OpenAI client not initialized. Cannot perform re-translation.")
                # Maybe send a notification?
                return

            new_translations = await self.translate_text_with_openai(
                text=new_text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )

            if new_translations is None:
                logger.error(f"Failed to re-translate content for edited message {after.id}.")
                # Optionally notify the user or admin
                # await after.channel.send(f"Sorry {after.author.mention}, failed to update translations for your edited message.", delete_after=15)
                return

            logger.info(f"Successfully received updated translation results for message {after.id}.")
            logger.debug(f"Updated translations: {new_translations}")

        # Handle errors during re-translation (simplified for brevity, could reuse handlers from on_message)
        except Exception as e:
            logger.error(f"Unexpected error during re-translation process for message {after.id}: {e}", exc_info=True)
            return

        # --- Update Translated Messages ---
        await self.update_translated_messages(after.id, new_translations)


    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """
        Translates text into multiple target languages using the openai library (connected to OpenRouter).
        (Logic remains the same as the previous version)
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized.")
            return None

        model_name = self.settings.get('translation_model', 'google/gemini-flash')

        # --- Construct Prompt ---
        system_prompt = f"""You are an expert multilingual translator. Translate the user's text from {source_lang} into the following languages: {', '.join(target_langs)}.
Respond ONLY with a valid JSON object containing the translations. The JSON object should have language codes (exactly as provided: {', '.join(target_langs)}) as keys and the corresponding translated text as string values.
Example format for targets {target_langs}:
{{
  "{target_langs[0]}": "translation for {target_langs[0]}",
  "{target_langs[1]}": "translation for {target_langs[1]}"
  // ... and so on for all target languages
}}
Ensure the output is nothing but the JSON object. Do not include any explanations, markdown formatting around the JSON, or introductory text. Preserve original formatting like markdown (e.g., bold, italics) within the translated strings where appropriate, but prioritize accurate translation. If the input text is empty or contains only whitespace, return an empty JSON object {{}}.
Target languages: {target_langs}"""

        user_prompt = text if text else ""
        content_str = None

        try:
            logger.debug(f"Sending request to OpenRouter (via OpenAI lib)...")
            logger.debug(f"Model: {model_name}, Target Languages: {target_langs}")

            response = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=1500
            )

            content_str = response.choices[0].message.content
            logger.debug(f"API response content (raw): {content_str[:500]}...")

            if not content_str:
                 logger.error("API response successful, but content is empty.")
                 return None

            # Clean up Markdown code blocks
            cleaned_content_str = content_str.strip()
            if cleaned_content_str.startswith("```json"):
                cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            if cleaned_content_str.startswith("```"):
                cleaned_content_str = cleaned_content_str[len("```"):].strip()
            if cleaned_content_str.endswith("```"):
                cleaned_content_str = cleaned_content_str[:-len("```")].strip()

            if not cleaned_content_str:
                 logger.error("Content is empty after cleaning Markdown symbols.")
                 logger.debug(f"Original content_str: {content_str}")
                 return None

            logger.debug(f"Cleaned content ready for parsing: {cleaned_content_str[:500]}...")
            translations = json.loads(cleaned_content_str)

            if not isinstance(translations, dict):
                logger.error(f"Parsed content is not a valid JSON object: {translations}")
                return None

            # Validate missing languages
            missing_langs = [lang for lang in target_langs if lang not in translations]
            if missing_langs:
                logger.warning(f"API response missing translations for some languages: {missing_langs}")
                for lang in missing_langs:
                    translations[lang] = f"[{lang.upper()} translation missing]"

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


    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, target_languages_map: dict):
        """
        Distribute translation results to target channels and store message mappings.
        """
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []
        sent_message_ids = {} # Temp dict to store {lang_code: message_id} for this distribution

        # Create a base embed
        base_embed = discord.Embed(color=discord.Color.blue())
        base_embed.set_author(name=f"{original_author.display_name} (from #{original_channel.name})", icon_url=original_author.display_avatar.url if original_author.display_avatar else None)
        base_embed.add_field(name="Original Message", value=f"[Click here to view]({original_message.jump_url})", inline=False)
        if original_message.attachments:
            attachment_links = [f"[{att.filename}]({att.url})" for i, att in enumerate(original_message.attachments) if i < 5]
            if len(original_message.attachments) > 5: attachment_links.append("...")
            if attachment_links: base_embed.add_field(name=f"Attachments ({len(original_message.attachments)})", value="\n".join(attachment_links), inline=False)

        # Inner function to send a single translation embed
        async def _send_translation(target_channel_id, lang_code, translated_text):
            # --- Added: Store sent message ID ---
            sent_msg = None
            try:
                target_channel = self.get_channel(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel):
                    embed_to_send = base_embed.copy()
                    embed_to_send.description = translated_text if translated_text and translated_text.strip() else "*(No text content or translation result is empty)*"
                    embed_to_send.set_footer(text=f"Original ({source_language.upper()}) → Target ({lang_code.upper()})")
                    # Send and get the message object
                    sent_msg = await target_channel.send(embed=embed_to_send)
                    logger.info(f"Sent translation ({lang_code}) to channel '{target_channel.name}' ({target_channel_id}) - Msg ID: {sent_msg.id}")
                    # Return the ID for mapping
                    return sent_msg.id
                else:
                    logger.warning(f"Could not find target channel ID {target_channel_id} or it's not a valid text channel.")
                    return None # Indicate failure
            except discord.errors.Forbidden: logger.error(f"Permission denied: Cannot send message to channel ID {target_channel_id} ({target_channel.name if target_channel else 'Unknown Channel'}).")
            except discord.errors.HTTPException as e: logger.error(f"HTTP error sending message to channel ID {target_channel_id} (Status: {e.status}): {e.text}")
            except Exception as e: logger.error(f"Unexpected error distributing translation to channel ID {target_channel_id}: {e}", exc_info=True)
            return None # Indicate failure

        # Create send tasks
        if translations:
            for lang_code, translated_text in translations.items():
                if lang_code in target_languages_map:
                    # Send to the first channel configured for this language
                    # (Modify if you need to send to multiple channels per language)
                    target_channel_id = target_languages_map[lang_code][0]
                    distribution_tasks.append(_send_translation(target_channel_id, lang_code, translated_text))
                else: logger.warning(f"Translation result included language code '{lang_code}' which is not in target_languages_map. Skipping.")
        elif not translations and original_message.attachments:
             logger.info(f"Syncing attachment links only to all target channels...")
             for lang_code, target_channel_ids in target_languages_map.items():
                 # Send to the first channel configured for this language
                 target_channel_id = target_channel_ids[0]
                 distribution_tasks.append(_send_translation(target_channel_id, lang_code, ""))
        else:
             logger.warning("No translation results and no attachments to distribute.")

        # Run tasks and collect results (sent message IDs)
        if distribution_tasks:
            results = await asyncio.gather(*distribution_tasks)
            # --- Added: Store the mapping ---
            successful_sends = {}
            lang_codes_in_order = list(translations.keys()) if translations else list(target_languages_map.keys()) # Get lang codes in order tasks were created

            for i, sent_id in enumerate(results):
                if sent_id: # Check if sending was successful (got an ID)
                    lang_code = lang_codes_in_order[i]
                    successful_sends[lang_code] = sent_id

            if successful_sends:
                 # Add/Update the map entry for the original message
                 self.message_map[original_message.id] = successful_sends
                 logger.info(f"Stored mapping for original message {original_message.id}: {successful_sends}")
                 # Optional: Prune old entries if map gets too large
                 if len(self.message_map) > self.message_map_max_size:
                     # Simple FIFO eviction
                     oldest_key = next(iter(self.message_map))
                     del self.message_map[oldest_key]
                     logger.info(f"Message map size exceeded {self.message_map_max_size}. Removed oldest entry: {oldest_key}")

            logger.info(f"Finished distributing all translations/attachments for original message {original_message.id}.")

    # --- Added: Function to update translated messages ---
    async def update_translated_messages(self, original_message_id: int, new_translations: dict):
        """
        Finds and edits translated messages based on the stored mapping.
        """
        if original_message_id not in self.message_map:
            logger.warning(f"Cannot update translations for {original_message_id}: No mapping found.")
            return

        translation_map = self.message_map[original_message_id]
        update_tasks = []

        async def _edit_translation(lang_code, translated_message_id, new_text):
            target_channel_id = None
            # Find the channel ID associated with this language and original message
            # This assumes the channel mapping hasn't changed drastically.
            # A more robust approach might store channel_id alongside message_id in the map.
            for cid_list in self.language_channel_map.get(lang_code, []):
                 # We need the specific channel where the message was sent.
                 # Fetching the message will give us the channel.
                 pass # We'll fetch the message below

            try:
                # Fetch the translated message object using its ID
                # This requires iterating through potential channels or storing channel ID in map
                # Let's try fetching from the first configured channel for that language
                target_channel_ids = self.language_channel_map.get(lang_code)
                if not target_channel_ids:
                    logger.warning(f"No channel configured for language {lang_code}, cannot fetch message {translated_message_id}")
                    return

                translated_message = None
                target_channel = None
                # Try fetching from the channels associated with the language
                for cid in target_channel_ids:
                    channel = self.get_channel(cid)
                    if channel:
                        try:
                            # fetch_message might raise NotFound or Forbidden
                            msg = await channel.fetch_message(translated_message_id)
                            if msg:
                                translated_message = msg
                                target_channel = channel
                                break # Found the message
                        except discord.NotFound:
                            continue # Try next channel
                        except discord.Forbidden:
                            logger.error(f"Permission error fetching message {translated_message_id} in channel {cid}")
                            continue # Try next channel
                if not translated_message:
                    logger.warning(f"Could not find translated message {translated_message_id} for lang {lang_code} in configured channels.")
                    # Remove this entry from the map as the message is likely deleted
                    if lang_code in self.message_map[original_message_id]:
                        del self.message_map[original_message_id][lang_code]
                    return

                # Check if the message has an embed to edit
                if translated_message.embeds:
                    original_embed = translated_message.embeds[0]
                    # Create a new embed based on the old one, only changing the description
                    new_embed = original_embed.copy()
                    new_embed.description = new_text if new_text and new_text.strip() else "*(No text content or translation result is empty)*"
                    # Edit the message with the new embed
                    await translated_message.edit(embed=new_embed)
                    logger.info(f"Successfully edited translated message {translated_message_id} in channel '{target_channel.name}' for language {lang_code}.")
                else:
                    logger.warning(f"Translated message {translated_message_id} has no embed to edit.")

            except discord.NotFound:
                logger.warning(f"Translated message {translated_message_id} not found (likely deleted). Removing from map.")
                if lang_code in self.message_map[original_message_id]:
                     del self.message_map[original_message_id][lang_code]
            except discord.Forbidden:
                logger.error(f"Permission error editing message {translated_message_id}. Bot might lack 'Manage Messages' permission or is trying to edit another user's message (shouldn't happen here).")
            except Exception as e:
                logger.error(f"Unexpected error editing translated message {translated_message_id}: {e}", exc_info=True)

        # Create tasks to edit each translated message
        for lang_code, translated_message_id in translation_map.items():
            if lang_code in new_translations:
                update_tasks.append(
                    _edit_translation(lang_code, translated_message_id, new_translations[lang_code])
                )
            else:
                 logger.warning(f"New translation missing for language {lang_code} during update for original message {original_message_id}.")


        # Run update tasks concurrently
        if update_tasks:
            await asyncio.gather(*update_tasks)
            logger.info(f"Finished updating translations for original message {original_message_id}.")


# --- Main Entry Point ---
if __name__ == "__main__":
    # --- Read config temporarily to check keys before full bot init ---
    temp_config = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: temp_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.critical(f"Error: Configuration file {CONFIG_FILE} not found. Please create it first.")
        TranslationBot(intents=discord.Intents.default(), config_path=CONFIG_FILE).create_example_config()
        exit(1)
    except yaml.YAMLError as e:
         logger.critical(f"Error: Failed to parse configuration file {CONFIG_FILE}: {e}")
         exit(1)

    # --- Read and validate necessary configurations ---
    api_keys = temp_config.get('api_keys', {})
    channels_config = temp_config.get('channels', {})
    discord_token = os.getenv('DISCORD_BOT_TOKEN') or api_keys.get('discord_token')
    openrouter_key = api_keys.get('openrouter_key')
    config_valid = True
    if not discord_token or discord_token == 'YOUR_DISCORD_BOT_TOKEN':
        logger.critical("Error: Valid Discord Bot Token not configured."); config_valid = False
    if not openrouter_key or openrouter_key == 'YOUR_OPENROUTER_API_KEY':
        logger.critical("Error: Valid OpenRouter API Key not configured."); config_valid = False
    if not channels_config:
        logger.critical("Error: No channels configured in config.yaml."); config_valid = False
    if not config_valid: exit(1)

    # --- Set up required Bot Intents ---
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    logger.info("Configuring Intents...")
    if not intents.message_content:
         logger.warning("Warning: Message Content Intent is not enabled.")

    # --- Initialize Bot instance with Intents ---
    bot = TranslationBot(intents=intents, config_path=CONFIG_FILE)

    logger.info("Starting Bot...")
    try:
        bot.run(discord_token, log_handler=None)
    except discord.LoginFailure: logger.critical("Login Failed: Invalid Discord Token.")
    except discord.PrivilegedIntentsRequired: logger.critical("Login Failed: Missing necessary Privileged Intents.")
    except Exception as e: logger.critical(f"Critical error during Bot startup: {e}", exc_info=True)
    finally: pass
