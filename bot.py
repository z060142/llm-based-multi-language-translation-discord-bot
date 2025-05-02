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

# --- 配置 ---
CONFIG_FILE = 'config.yaml'
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('TranslationBot')

# OpenRouter 的 OpenAI 相容基礎 URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- 核心 Bot 類別 ---
class TranslationBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, config_path: str):
        super().__init__(intents=intents)
        self.config_path = config_path
        self.channel_language_map = {}
        self.language_channel_map = {}
        self.api_keys = {}
        self.settings = {
            'translation_model': 'google/gemini-flash',
            'max_retries': 2,
            'retry_delay': 3
        }
        self.openai_client: AsyncOpenAI | None = None

    async def setup_hook(self) -> None:
        self.load_config()
        logger.info(f'設定載入完成。監聽 {len(self.channel_language_map)} 個頻道。')
        logger.info(f"將使用模型 '{self.settings.get('translation_model')}' 進行翻譯。")
        openrouter_key = self.api_keys.get('openrouter_key')
        if openrouter_key and openrouter_key != 'YOUR_OPENROUTER_API_KEY':
            self.openai_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_key,
            )
            logger.info("OpenAI 客戶端已為 OpenRouter 初始化。")
        else:
            logger.error("未能初始化 OpenAI 客戶端：缺少有效的 OpenRouter API Key。")


    async def close(self) -> None:
        await super().close()
        logger.info("Bot 正在關閉。")

    def load_config(self):
        """從 YAML 檔案載入設定"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if not config_data:
                    logger.error(f"設定檔 '{self.config_path}' 是空的。")
                    self.create_example_config()
                    logger.critical("已建立範例設定檔，請填入您的資訊後重新啟動 Bot。")
                    return
                # 載入頻道設定
                if 'channels' in config_data and isinstance(config_data['channels'], dict):
                    self.channel_language_map = {}
                    self.language_channel_map = {}
                    for lang_code, channel_ids in config_data['channels'].items():
                        if not isinstance(channel_ids, list): channel_ids = [channel_ids]
                        if lang_code not in self.language_channel_map: self.language_channel_map[lang_code] = []
                        for channel_id in channel_ids:
                            try:
                                ch_id_int = int(channel_id)
                                if ch_id_int in self.channel_language_map: logger.warning(f"頻道 ID {ch_id_int} 被重複設定於語言 '{self.channel_language_map[ch_id_int]}' 和 '{lang_code}'。將使用後者 '{lang_code}'。")
                                self.channel_language_map[ch_id_int] = lang_code
                                self.language_channel_map[lang_code].append(ch_id_int)
                            except ValueError: logger.warning(f"設定檔中發現無效的頻道 ID '{channel_id}' (語言: {lang_code})，已跳過。")
                    logger.info(f"成功載入 {len(self.channel_language_map)} 個頻道設定。")
                else: logger.warning(f"設定檔 '{self.config_path}' 中缺少 'channels' 區塊或格式不正確。")
                # 載入 API 金鑰
                if 'api_keys' in config_data and isinstance(config_data['api_keys'], dict):
                    self.api_keys = config_data['api_keys']
                    logger.info("成功載入 API 金鑰設定。")
                else: logger.warning(f"設定檔 '{self.config_path}' 中缺少 'api_keys' 區塊或格式不正確。")
                # 載入其他設定
                if 'settings' in config_data and isinstance(config_data['settings'], dict):
                    self.settings['translation_model'] = config_data['settings'].get('translation_model', self.settings['translation_model'])
                    self.settings['max_retries'] = config_data['settings'].get('max_retries', self.settings['max_retries'])
                    self.settings['retry_delay'] = config_data['settings'].get('retry_delay', self.settings['retry_delay'])
                    logger.info("成功載入其他設定。")
                else: logger.info("未找到 'settings' 區塊，將使用預設設定。")
        except FileNotFoundError:
            logger.error(f"找不到設定檔: {self.config_path}。")
            self.create_example_config()
            logger.critical("已建立範例設定檔，請填入您的 Discord Token、OpenRouter Key 和頻道 ID 後重新啟動 Bot。")
        except yaml.YAMLError as e: logger.error(f"解析設定檔 '{self.config_path}' 時發生錯誤: {e}")
        except Exception as e: logger.error(f"載入設定時發生未預期的錯誤: {e}", exc_info=True)

    def create_example_config(self):
        """建立一個範例設定檔"""
        default_config = {
            'channels': {'en': [123456789012345678], 'zh-TW': [987654321098765432], 'ja': [555555555555555555]},
            'api_keys': {'discord_token': 'YOUR_DISCORD_BOT_TOKEN', 'openrouter_key': 'YOUR_OPENROUTER_API_KEY'},
            'settings': {'translation_model': 'google/gemini-flash', 'max_retries': 2, 'retry_delay': 3}
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f: yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            logger.info(f"已建立範例設定檔 '{self.config_path}'。請填入您的資訊。")
        except Exception as e: logger.error(f"建立範例設定檔 '{self.config_path}' 時發生錯誤: {e}")

    async def on_ready(self):
        """Bot 準備就緒"""
        logger.info(f'以 {self.user} 登入成功！ (ID: {self.user.id})')
        logger.info('機器人已準備就緒。')

    async def on_message(self, message: discord.Message):
        """處理訊息事件"""
        if message.author == self.user or message.author.bot: return
        source_channel_id = message.channel.id
        if source_channel_id not in self.channel_language_map: return
        if not message.content and not message.attachments:
            logger.debug(f"來自頻道 {message.channel.name} 的訊息內容和附件皆為空，忽略。")
            return

        source_language = self.channel_language_map.get(source_channel_id)
        if not source_language:
            logger.warning(f"內部錯誤：在 channel_language_map 中找不到頻道 ID {source_channel_id} 的語言設定。")
            return

        target_languages_map = {}
        target_language_codes = []
        for lang_code, channel_ids in self.language_channel_map.items():
            if lang_code != source_language:
                target_languages_map[lang_code] = channel_ids
                target_language_codes.append(lang_code)

        if not target_languages_map:
            logger.info(f"頻道 {message.channel.name} ({source_language}) 的訊息沒有其他目標語言頻道可同步。")
            return

        text_to_translate = message.content
        if not text_to_translate and message.attachments:
             logger.info(f"訊息 {message.id} 僅包含附件，將僅同步附件連結。")
             await self.distribute_translations(message, source_language, {}, target_languages_map)
             return

        logger.info(f"偵測到來自頻道 '{message.channel.name}' ({source_channel_id}, 語言: {source_language}) 的訊息，作者: {message.author.display_name}")
        log_content = text_to_translate[:100] + ('...' if len(text_to_translate) > 100 else '')
        logger.info(f"準備將內容翻譯至語言: {target_language_codes}")
        logger.debug(f"待翻譯內容預覽: '{log_content}'")

        # --- 執行翻譯 ---
        try:
            if not self.openai_client:
                logger.error("OpenAI 客戶端未初始化，無法進行翻譯。請檢查 API Key 設定。")
                raise RuntimeError("OpenAI client not initialized")

            translations = await self.translate_text_with_openai(
                text=text_to_translate,
                source_lang=source_language,
                target_langs=target_language_codes
            )

            if translations is None:
                await message.channel.send(f"抱歉 {message.author.mention}，翻譯時遇到問題，無法完成同步。請檢查日誌獲取詳細資訊。", delete_after=15)
                return

            logger.info(f"成功從 API 獲取翻譯結果。")
            logger.debug(f"翻譯結果: {translations}")

        except openai.AuthenticationError:
             logger.error("OpenAI API 驗證失敗：無效的 API Key 或權限不足。")
             await message.channel.send(f"抱歉 {message.author.mention}，翻譯服務驗證失敗，請聯繫管理員。", delete_after=15)
             return
        except openai.RateLimitError:
            logger.warning("OpenAI API 速率限制：請求過於頻繁。")
            await message.channel.send(f"抱歉 {message.author.mention}，翻譯請求過於頻繁，請稍後再試。", delete_after=15)
            return
        except openai.APIConnectionError as e:
            logger.error(f"無法連接到 OpenAI API: {e}")
            await message.channel.send(f"抱歉 {message.author.mention}，無法連接到翻譯服務。", delete_after=15)
            return
        except openai.APIError as e:
             logger.error(f"OpenAI API 返回錯誤: {e}")
             await message.channel.send(f"抱歉 {message.author.mention}，翻譯服務遇到錯誤。", delete_after=15)
             return
        except Exception as e:
            logger.error(f"翻譯過程中發生未預期的錯誤: {e}", exc_info=True)
            await message.channel.send(f"抱歉 {message.author.mention}，翻譯過程中發生內部錯誤。", delete_after=15)
            return

        # --- 分發翻譯結果 ---
        await self.distribute_translations(message, source_language, translations, target_languages_map)


    async def translate_text_with_openai(self, text: str, source_lang: str, target_langs: list[str]) -> dict | None:
        """
        使用 openai 函式庫 (連接 OpenRouter) 將文字翻譯成多種目標語言。
        """
        if not self.openai_client:
            logger.error("OpenAI 客戶端未初始化。")
            return None

        model_name = self.settings.get('translation_model', 'google/gemini-flash')

        # --- 建構 Prompt ---
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
        content_str = None # <<< 初始化 content_str

        try:
            logger.debug(f"向 OpenRouter (via OpenAI lib) 發送請求...")
            logger.debug(f"模型: {model_name}, 目標語言: {target_langs}")

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

            content_str = response.choices[0].message.content # <<< 獲取回應內容
            logger.debug(f"API 回應內容 (原始): {content_str[:500]}...")

            if not content_str:
                 logger.error("API 回應成功，但 content 為空。")
                 return None

            # --- 新增：清理 Markdown 程式碼區塊 ---
            cleaned_content_str = content_str.strip()
            if cleaned_content_str.startswith("```json"):
                cleaned_content_str = cleaned_content_str[len("```json"):].strip()
            if cleaned_content_str.startswith("```"): # 處理沒有指定語言的情況
                cleaned_content_str = cleaned_content_str[len("```"):].strip()
            if cleaned_content_str.endswith("```"):
                cleaned_content_str = cleaned_content_str[:-len("```")].strip()
            # --- 清理結束 ---

            # 檢查清理後是否為空
            if not cleaned_content_str:
                 logger.error("清理 Markdown 符號後，內容為空。")
                 logger.debug(f"原始 content_str: {content_str}")
                 return None


            logger.debug(f"清理後的準備解析內容: {cleaned_content_str[:500]}...") # 記錄清理後的內容
            translations = json.loads(cleaned_content_str) # <<< 使用清理後的字串解析

            if not isinstance(translations, dict):
                logger.error(f"解析後的內容不是有效的 JSON 物件: {translations}")
                return None

            # 驗證是否包含所有目標語言
            missing_langs = [lang for lang in target_langs if lang not in translations]
            if missing_langs:
                logger.warning(f"API 回應缺少部分語言的翻譯: {missing_langs}")
                for lang in missing_langs:
                    translations[lang] = f"[{lang.upper()} 翻譯缺失]"

            filtered_translations = {k: v for k, v in translations.items() if k in target_langs}
            return filtered_translations

        except json.JSONDecodeError as e:
            logger.error(f"解析 API 回應 JSON 時失敗: {e}")
            # 確保 content_str 被賦值過才記錄
            original_content_to_log = content_str if content_str is not None else "[未能獲取原始回應內容]"
            logger.error(f"原始回應內容: {original_content_to_log}")
            return None
        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"處理 API 回應結構時出錯: {e}")
             # 確保 response 被賦值過才記錄
             response_to_log = response if 'response' in locals() else "[未能獲取回應物件]"
             logger.debug(f"完整 API 回應物件: {response_to_log}")
             return None
        except Exception as e:
            logger.error(f"使用 OpenAI 函式庫翻譯時發生未預期錯誤: {e}", exc_info=True)
            return None


    async def distribute_translations(self, original_message: discord.Message, source_language: str, translations: dict, target_languages_map: dict):
        """將翻譯結果分發到目標頻道"""
        original_author = original_message.author
        original_channel = original_message.channel
        distribution_tasks = []
        base_embed = discord.Embed(color=discord.Color.blue())
        base_embed.set_author(name=f"{original_author.display_name} (來自 #{original_channel.name})", icon_url=original_author.display_avatar.url if original_author.display_avatar else None)
        base_embed.add_field(name="原文連結", value=f"[點此查看]({original_message.jump_url})", inline=False)
        if original_message.attachments:
            attachment_links = [f"[{att.filename}]({att.url})" for i, att in enumerate(original_message.attachments) if i < 5]
            if len(original_message.attachments) > 5: attachment_links.append("...")
            if attachment_links: base_embed.add_field(name=f"附件 ({len(original_message.attachments)})", value="\n".join(attachment_links), inline=False)

        async def _send_translation(target_channel_id, lang_code, translated_text):
            try:
                target_channel = self.get_channel(target_channel_id)
                if target_channel and isinstance(target_channel, discord.TextChannel):
                    embed_to_send = base_embed.copy()
                    embed_to_send.description = translated_text if translated_text and translated_text.strip() else "*(無文字內容或翻譯結果為空)*"
                    embed_to_send.set_footer(text=f"原文 ({source_language.upper()}) → 目標 ({lang_code.upper()})")
                    await target_channel.send(embed=embed_to_send)
                    logger.info(f"已將翻譯 ({lang_code}) 發送至頻道 '{target_channel.name}' ({target_channel_id})")
                else: logger.warning(f"找不到目標頻道 ID {target_channel_id} 或該頻道不是有效的文字頻道。")
            except discord.errors.Forbidden: logger.error(f"權限不足，無法發送訊息至頻道 ID {target_channel_id} ({target_channel.name if target_channel else '未知頻道'})。")
            except discord.errors.HTTPException as e: logger.error(f"發送訊息至頻道 ID {target_channel_id} 時發生 HTTP 錯誤 (狀態碼: {e.status}): {e.text}")
            except Exception as e: logger.error(f"分發翻譯至頻道 ID {target_channel_id} 時發生未預期錯誤: {e}", exc_info=True)

        if translations:
            for lang_code, translated_text in translations.items():
                if lang_code in target_languages_map:
                    for target_channel_id in target_languages_map[lang_code]: distribution_tasks.append(_send_translation(target_channel_id, lang_code, translated_text))
                else: logger.warning(f"翻譯結果中包含未在 target_languages_map 中的語言代碼 '{lang_code}'，已忽略。")
        elif not translations and original_message.attachments:
             logger.info(f"僅同步附件連結至所有目標頻道...")
             for lang_code, target_channel_ids in target_languages_map.items():
                 for target_channel_id in target_channel_ids: distribution_tasks.append(_send_translation(target_channel_id, lang_code, ""))
        else: logger.warning("沒有翻譯結果且沒有附件，無需分發。")

        if distribution_tasks:
            await asyncio.gather(*distribution_tasks)
            logger.info(f"已完成對原始訊息 {original_message.id} 的所有翻譯/附件分發。")

# --- 主程式入口 (與上一版相同) ---
if __name__ == "__main__":
    temp_config = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: temp_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.critical(f"錯誤：找不到設定檔 {CONFIG_FILE}。請先建立設定檔。")
        TranslationBot(intents=discord.Intents.default(), config_path=CONFIG_FILE).create_example_config()
        exit(1)
    except yaml.YAMLError as e:
         logger.critical(f"錯誤：解析設定檔 {CONFIG_FILE} 失敗: {e}")
         exit(1)

    api_keys = temp_config.get('api_keys', {})
    channels_config = temp_config.get('channels', {})
    discord_token = os.getenv('DISCORD_BOT_TOKEN') or api_keys.get('discord_token')
    openrouter_key = api_keys.get('openrouter_key')
    config_valid = True
    if not discord_token or discord_token == 'YOUR_DISCORD_BOT_TOKEN': logger.critical("錯誤：未設定有效的 Discord Bot Token..."); config_valid = False
    if not openrouter_key or openrouter_key == 'YOUR_OPENROUTER_API_KEY': logger.critical("錯誤：未設定有效的 OpenRouter API Key..."); config_valid = False
    if not channels_config: logger.critical("錯誤：未在 config.yaml 中設定任何頻道。"); config_valid = False
    if not config_valid: exit(1)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    logger.info("正在設定 Intents...")
    if not intents.message_content: logger.warning("警告：Message Content Intent 未啟用...")

    bot = TranslationBot(intents=intents, config_path=CONFIG_FILE)
    logger.info("正在啟動 Bot...")
    try:
        bot.run(discord_token, log_handler=None)
    except discord.LoginFailure: logger.critical("登入失敗：無效的 Discord Token。")
    except discord.PrivilegedIntentsRequired: logger.critical("登入失敗：缺少必要的 Privileged Intents...")
    except Exception as e: logger.critical(f"啟動 Bot 時發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally: pass
