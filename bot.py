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
            'translation_model': 'google/gemini-flash', # Use a faster and cheaper model as default
            'max_retries': 2, # Max retries for API requests
            'retry_delay': 3  # Delay between retries in seconds
        }
        self.openai_client: AsyncOpenAI | None = None # OpenAI client

    async def setup_hook(self) -> None:
        """
        Asynchronous setup hook called before the Bot starts.
        Suitable for loading configurations, initializing resources, etc.
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
        # OpenAI client usually doesn't require manual session closing
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
                                # Warn if a channel ID is configured for multiple languages
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
                    # Update using get() to keep defaults if not specified in the config
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
        """Handle message events."""
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
            # This should theoretically not happen due to the check above
            logger.warning(f"Internal error: Could not find language setting for channel ID {source_channel_id} in channel_language_map.")
            return

        # Determine target languages and channels, excluding the source language
        target_languages_map = {}
        target_language_codes = []
        for lang_code, channel_ids in self.language_channel_map.items():
            if lang_code != source_language:
                target_languages_map[lang_code] = channel_ids
                target_language_codes.append(lang_code)

        # If no other languages are configured, do nothing
        if not target_languages_map:
            logger.info(f"No target language channels configured for messages from channel {message.channel.name} ({source_language}).")
            return

        # Get text to translate
        text_to_translate = message.content
        # If no text but attachments exist, only sync attachments
        if not text_to_translate and message.attachments:
             logger.info(f"Message {message.id} contains only attachments. Syncing attachment links only.")
             # Call distribution directly with an empty translations dictionary
             await self.distribute_translations(message, source_language, {}, target_languages_map)
             return

        # Log received message details
        logger.info(f"Detected message from channel '{message.channel.name}' ({source_channel_id}, Language: {source_language}), Author: {message.author.display_name}")
        log_content = text_to_translate[:100] + ('...' if len(text_to_translate) > 100 else '') # Limit log length
        logger.info(f"Preparing to translate content to languages: {target_language_codes}")
        logger.debug(f"Content preview for translation: '{log_content}'")

        # --- Perform Translation ---
        try:
            # Ensure OpenAI client is initialized
            if not self.openai_client:
                logger.error("OpenAI client not initialized. Cannot perform translation. Please check API Key settings.")
                raise RuntimeError("OpenAI client not initialized")

            # Call the translation function
            translations = await self.translate_text_with_openai(
                text=text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )

            # Handle translation failure
            if translations is None:
                # Detailed error already logged in translate_text_with_openai
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
        await self.distribute_translations(message, source_language, translations, target_languages_map)


    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """
        Translates text into multiple target languages using the openai library (connected to OpenRouter).
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
        content_str = None # Initialize content_str

        try:
            logger.debug(f"Sending request to OpenRouter (via OpenAI lib)...")
            logger.debug(f"Model: {model_name}, Target Languages: {target_langs}")

            # Call OpenAI compatible API endpoint
            response = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # Request JSON output
                temperature=0.5, # Adjust temperature for more consistent output
                max_tokens=1500 # Set max tokens as needed
            )

            content_str = response.choices[0].message.content # Get response content
            logger.debug(f"API response content (raw): {content_str[:500]}...") # Log first 500 chars

            if not content_str:
                 logger.error("API response successful, but content is empty.")
                 return None

            # --- Added: Clean up Markdown code blocks ---
            cleaned_content_str = content_str.strip() # Remove leading/trailing whitespace
            if cleaned_content_str.startswith("```json"):
                # Remove ```json prefix
                cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            elif cleaned_content_str.startswith("```"): # Handle ``` without language specified
                cleaned_content_str = cleaned_content_str[len("```"):].strip()

            # Remove ``` suffix
            if cleaned_content_str.endswith("```"):
                cleaned_content_str = cleaned_content_str[:-len("```")].strip()
            # --- Cleaning finished ---

            # Check if empty after cleaning
            if not cleaned_content_str:
                 logger.error("Content is empty after cleaning Markdown symbols.")
                 logger.debug(f"Original content_str: {content_str}")
                 return None


            logger.debug(f"Cleaned content ready for parsing: {cleaned_content_str[:500]}...") # Log cleaned content
            translations = json.loads(cleaned_content_str) # Parse using the cleaned string

            # Validate if the result is a dictionary
            if not isinstance(translations, dict):
                logger.error(f"Parsed content is not a valid JSON object: {translations}")
                return None

            # Validate if all target languages are present (optional but recommended)
            missing_langs = [lang for lang in target_langs if lang not in translations]
            if missing_langs:
                logger.warning(f"API response missing translations for some languages: {missing_langs}")
                # Fill missing translations with a placeholder
                for lang in missing_langs:
                    translations[lang] = f"[{lang.upper()} translation missing]"

            # Filter out any keys not in target languages (if the model returned extra)
            filtered_translations = {k: v for k, v in translations.items() if k in target_langs}
            return filtered_translations

        except json.JSONDecodeError as e:
            # Log error if JSON parsing fails
            logger.error(f"Failed to parse API response JSON: {e}")
            # Ensure content_str was assigned before logging
            original_content_to_log = content_str if content_str is not None else "[Could not get raw response content]"
            logger.error(f"Raw response content: {original_content_to_log}")
            return None
        except (AttributeError, IndexError, TypeError) as e:
             # Log error if response structure is unexpected
             logger.error(f"Error processing API response structure: {e}")
             # Ensure response was assigned before logging
             response_to_log = response if 'response' in locals() else "[Could not get response object]"
             logger.debug(f"Full API response object: {response_to_log}")
             return None
        # Catch other unexpected errors during translation
        except Exception as e:
            logger.error(f"Unexpected error during translation using OpenAI library: {e}", exc_info=True)
            return None


    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, target_languages_map: dict):
        """Distribute translation results to target channels."""
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []

        # Create a base embed with author, original link, and attachments
        base_embed = discord.Embed(color=discord.Color.blue())
        base_embed.set_author(name=f"{original_author.display_name} (from #{original_channel.name})", icon_url=original_author.display_avatar.url if original_author.display_avatar else None)
        base_embed.add_field(name="Original Message", value=f"[Click here to view]({original_message.jump_url})", inline=False)
        if original_message.attachments:
            attachment_links = [f"[{att.filename}]({att.url})" for i, att in enumerate(original_message.attachments) if i < 5]
            if len(original_message.attachments) > 5: attachment_links.append("...")
            if attachment_links: base_embed.add_field(name=f"Attachments ({len(original_message.attachments)})", value="\n".join(attachment_links), inline=False)

        # Inner function to send a single translation embed
        async def _send_translation(target_channel_id, lang_code, translated_text):
            try:
                target_channel = self.get_channel(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel):
                    # Copy the base embed and add translation details
                    embed_to_send = base_embed.copy()
                    embed_to_send.description = translated_text if translated_text and translated_text.strip() else "*(No text content or translation result is empty)*"
                    embed_to_send.set_footer(text=f"Original ({source_language.upper()}) → Target ({lang_code.upper()})")
                    # Send the embed
                    await target_channel.send(embed=embed_to_send)
                    logger.info(f"Sent translation ({lang_code}) to channel '{target_channel.name}' ({target_channel_id})")
                else: logger.warning(f"Could not find target channel ID {target_channel_id} or it's not a valid text channel.")
            except discord.errors.Forbidden: logger.error(f"Permission denied: Cannot send message to channel ID {target_channel_id} ({target_channel.name if target_channel else 'Unknown Channel'}).")
            except discord.errors.HTTPException as e: logger.error(f"HTTP error sending message to channel ID {target_channel_id} (Status: {e.status}): {e.text}")
            except Exception as e: logger.error(f"Unexpected error distributing translation to channel ID {target_channel_id}: {e}", exc_info=True)

        # Create send tasks based on translations or attachments
        if translations: # If there are text translations
            for lang_code, translated_text in translations.items():
                if lang_code in target_languages_map:
                    for target_channel_id in target_languages_map[lang_code]:
                        distribution_tasks.append(_send_translation(target_channel_id, lang_code, translated_text))
                else: logger.warning(f"Translation result included language code '{lang_code}' which is not in target_languages_map. Skipping.")
        elif not translations and original_message.attachments: # If only attachments need syncing
             logger.info(f"Syncing attachment links only to all target channels...")
             for lang_code, target_channel_ids in target_languages_map.items():
                 for target_channel_id in target_channel_ids:
                     # Send the base embed with an empty description
                     distribution_tasks.append(_send_translation(target_channel_id, lang_code, ""))
        else: # No translations and no attachments
             logger.warning("No translation results and no attachments to distribute.")

        # Run all distribution tasks concurrently
        if distribution_tasks:
            await asyncio.gather(*distribution_tasks)
            logger.info(f"Finished distributing all translations/attachments for original message {original_message.id}.")

# --- Main Entry Point ---
if __name__ == "__main__":
    # --- Read config temporarily to check keys before full bot init ---
    temp_config = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: temp_config = yaml.safe_load(f) or {} # Ensure dict even if empty
    except FileNotFoundError:
        logger.critical(f"Error: Configuration file {CONFIG_FILE} not found. Please create it first.")
        # Attempt to create example, but exit anyway as token is missing
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
        logger.critical("Error: Valid Discord Bot Token not configured. Please set environment variable or configure in config.yaml."); config_valid = False
    if not openrouter_key or openrouter_key == 'YOUR_OPENROUTER_API_KEY':
        logger.critical("Error: Valid OpenRouter API Key not configured. Please configure in config.yaml."); config_valid = False
    if not channels_config:
        logger.critical("Error: No channels configured in config.yaml."); config_valid = False
    if not config_valid: exit(1) # Exit if config is invalid

    # --- Set up required Bot Intents ---
    # Intents must be configured before initializing the Client
    intents = discord.Intents.default()
    intents.message_content = True # Required to read message content
    intents.guilds = True          # Required for guild-related events/info
    intents.messages = True        # Required for message events (on_message)

    logger.info("Configuring Intents...")
    if not intents.message_content:
         # This warning is important for discord.py v2+
         logger.warning("Warning: Message Content Intent is not enabled. Bot might not be able to read message content. Please enable it in the Discord Developer Portal.")

    # --- Initialize Bot instance with Intents ---
    # Intents are now passed during initialization
    bot = TranslationBot(intents=intents, config_path=CONFIG_FILE)

    logger.info("Starting Bot...")
    try:
        # Run the Bot using the token
        # The bot instance will call load_config internally via setup_hook
        bot.run(discord_token, log_handler=None) # Use None to disable discord.py's default logging handler if we configured our own
    except discord.LoginFailure:
        logger.critical("Login Failed: Invalid Discord Token.")
    except discord.PrivilegedIntentsRequired:
         logger.critical("Login Failed: Missing necessary Privileged Intents (e.g., Message Content). Please enable them in the Discord Developer Portal.")
    except Exception as e:
        logger.critical(f"Critical error during Bot startup: {e}", exc_info=True)
    finally:
        # The bot.run() method handles cleanup, including calling bot.close()
        pass
