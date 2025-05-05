#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import os
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import yaml
import os
import subprocess
import threading
import queue
from enum import Enum
import re

class BotStatus(Enum):
    STOPPED = "Stopped"
    RUNNING = "Running"
    ERROR = "Error"

class ChannelDialog:
    """Channel configuration dialog"""
    def __init__(self, parent, title="Add Channel", default_values=None):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        # Create input fields
        ttk.Label(self.dialog, text="Language Code:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.lang_var = tk.StringVar(value=default_values[0] if default_values else "")
        lang_entry = ttk.Entry(self.dialog, textvariable=self.lang_var, width=30)
        lang_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.dialog, text="Channel ID:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.channel_id_var = tk.StringVar(value=default_values[1] if default_values else "")
        channel_id_entry = ttk.Entry(self.dialog, textvariable=self.channel_id_var, width=30)
        channel_id_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.dialog, text="Channel Type:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.type_var = tk.StringVar(value=default_values[2] if default_values else "standard")
        type_combo = ttk.Combobox(self.dialog, textvariable=self.type_var, 
                                  values=["standard", "read_only", "write_only"], 
                                  state="readonly")
        type_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ok_btn = ttk.Button(btn_frame, text="OK", command=self.ok)
        ok_btn.pack(side="left", padx=5)
        
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side="left", padx=5)
        
        # Bind keys
        self.dialog.bind('<Return>', lambda e: self.ok())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        
        # Center dialog
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # Focus on first input
        lang_entry.focus_set()
    
    def ok(self):
        lang = self.lang_var.get().strip()
        channel_id = self.channel_id_var.get().strip()
        
        if not lang or not channel_id:
            messagebox.showerror("Error", "Please fill in all fields!")
            return
        
        try:
            channel_id_int = int(channel_id)
        except ValueError:
            messagebox.showerror("Error", "Channel ID must be a number!")
            return
        
        self.result = (lang, channel_id_int, self.type_var.get())
        self.dialog.destroy()
    
    def cancel(self):
        self.dialog.destroy()

class ServiceDialog:
    """Service configuration dialog"""
    def __init__(self, parent, title="Add Service", default_values=None):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x500")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        # Service provider
        ttk.Label(self.dialog, text="Service Provider:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.provider_var = tk.StringVar(value=default_values.get('provider') if default_values else "openai")
        provider_combo = ttk.Combobox(self.dialog, textvariable=self.provider_var, 
                                      values=["openai", "azure", "ollama"], 
                                      state="readonly")
        provider_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Base URL
        ttk.Label(self.dialog, text="Base URL:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.base_url_var = tk.StringVar(value=default_values.get('base_url') if default_values else "")
        base_url_entry = ttk.Entry(self.dialog, textvariable=self.base_url_var, width=40)
        base_url_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Model
        ttk.Label(self.dialog, text="Model:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.model_var = tk.StringVar(value=default_values.get('model') if default_values else "")
        model_entry = ttk.Entry(self.dialog, textvariable=self.model_var, width=40)
        model_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # API Key
        ttk.Label(self.dialog, text="API Key:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.api_key_var = tk.StringVar(value=default_values.get('api_key') if default_values else "")
        api_key_entry = ttk.Entry(self.dialog, textvariable=self.api_key_var, width=40)
        api_key_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Azure specific fields
        self.azure_frame = ttk.LabelFrame(self.dialog, text="Azure-specific Options", padding="5")
        self.azure_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.azure_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(self.azure_frame, text="API Version:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.api_version_var = tk.StringVar(value=default_values.get('api_version') if default_values else "")
        api_version_entry = ttk.Entry(self.azure_frame, textvariable=self.api_version_var, width=30)
        api_version_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.azure_frame, text="Deployment Name:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.deployment_name_var = tk.StringVar(value=default_values.get('deployment_name') if default_values else "")
        deployment_name_entry = ttk.Entry(self.azure_frame, textvariable=self.deployment_name_var, width=30)
        deployment_name_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Ollama specific fields
        self.ollama_frame = ttk.LabelFrame(self.dialog, text="Ollama-specific Options", padding="5")
        self.ollama_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.ollama_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(self.ollama_frame, text="Timeout (seconds):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.timeout_var = tk.StringVar(value=str(default_values.get('timeout', '90')) if default_values else "90")
        timeout_entry = ttk.Entry(self.ollama_frame, textvariable=self.timeout_var, width=10)
        timeout_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        ok_btn = ttk.Button(btn_frame, text="OK", command=self.ok)
        ok_btn.pack(side="left", padx=5)
        
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side="left", padx=5)
        
        # Bind events
        self.dialog.bind('<Return>', lambda e: self.ok())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        provider_combo.bind('<<ComboboxSelected>>', self.update_fields_visibility)
        
        # Initialize field visibility
        self.update_fields_visibility()
        
        # Center dialog
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def update_fields_visibility(self, event=None):
        provider = self.provider_var.get()
        
        if provider == "azure":
            self.azure_frame.grid()
            self.ollama_frame.grid_remove()
        elif provider == "ollama":
            self.azure_frame.grid_remove()
            self.ollama_frame.grid()
        else:
            self.azure_frame.grid_remove()
            self.ollama_frame.grid_remove()
    
    def ok(self):
        provider = self.provider_var.get()
        base_url = self.base_url_var.get().strip()
        model = self.model_var.get().strip()
        api_key = self.api_key_var.get().strip()
        
        if not provider or not base_url or not model:
            messagebox.showerror("Error", "Please fill in all required fields!")
            return
        
        result = {
            'provider': provider,
            'base_url': base_url,
            'model': model
        }
        
        if provider != "ollama":
            if not api_key:
                messagebox.showerror("Error", f"{provider} requires an API Key!")
                return
            result['api_key'] = api_key
        else:
            result['api_key'] = None
        
        if provider == "azure":
            result['api_version'] = self.api_version_var.get().strip()
            result['deployment_name'] = self.deployment_name_var.get().strip()
        elif provider == "ollama":
            try:
                timeout = int(self.timeout_var.get())
                result['timeout'] = timeout
            except ValueError:
                messagebox.showerror("Error", "Timeout must be an integer!")
                return
        
        self.result = result
        self.dialog.destroy()
    
    def cancel(self):
        self.dialog.destroy()

class TranslationBotUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Discord Translation Bot Configuration Tool")
        self.root.geometry("900x800")
        
        # Initialize variables
        self.config_data = {}
        self.bot_process = None
        self.log_queue = queue.Queue()
        self.current_status = BotStatus.STOPPED
        
        # Create main interface
        self.create_widgets()
        
        # Load configuration
        self.load_config()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Periodically check logs
        self.check_log_queue()
    
    def create_widgets(self):
        """Create main UI components"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create notebook tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill="both", expand=True)
        
        # === Basic Settings Tab ===
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Settings")
        
        # Discord Token
        token_frame = ttk.LabelFrame(basic_frame, text="Discord Token", padding="5")
        token_frame.pack(fill="x", padx=5, pady=5)
        self.token_var = tk.StringVar()
        token_entry = ttk.Entry(token_frame, textvariable=self.token_var, show="*", width=50)
        token_entry.pack(fill="x", padx=5, pady=5)
        
        # Translation Settings
        translation_frame = ttk.LabelFrame(basic_frame, text="Translation Settings", padding="5")
        translation_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(translation_frame, text="Translation Tone:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.tone_var = tk.StringVar()
        tone_entry = ttk.Entry(translation_frame, textvariable=self.tone_var, width=30)
        tone_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(translation_frame, text="Special Instructions:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.special_var = tk.StringVar()
        special_entry = ttk.Entry(translation_frame, textvariable=self.special_var, width=30)
        special_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(translation_frame, text="Text Separation Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.threshold_var = tk.StringVar(value="1200")
        threshold_entry = ttk.Entry(translation_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # === Channel Configuration Tab ===
        channels_frame = ttk.Frame(notebook)
        notebook.add(channels_frame, text="Channel Configuration")
        
        # Channel list
        channel_list_frame = ttk.Frame(channels_frame)
        channel_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create Treeview
        self.channel_tree = ttk.Treeview(channel_list_frame, columns=('lang', 'channel_id', 'type'), show='headings')
        self.channel_tree.heading('lang', text='Language')
        self.channel_tree.heading('channel_id', text='Channel ID')
        self.channel_tree.heading('type', text='Type')
        
        self.channel_tree.column('lang', width=80)
        self.channel_tree.column('channel_id', width=200)
        self.channel_tree.column('type', width=100)
        
        # Scrollbar
        channel_scrollbar = ttk.Scrollbar(channel_list_frame, orient="vertical", command=self.channel_tree.yview)
        self.channel_tree.configure(yscrollcommand=channel_scrollbar.set)
        
        self.channel_tree.pack(side="left", fill="both", expand=True)
        channel_scrollbar.pack(side="right", fill="y")
        
        # Channel operation buttons
        channel_btn_frame = ttk.Frame(channels_frame)
        channel_btn_frame.pack(fill="x", padx=5, pady=5)
        
        add_channel_btn = ttk.Button(channel_btn_frame, text="Add Channel", command=self.add_channel)
        add_channel_btn.pack(side="left", padx=5)
        
        edit_channel_btn = ttk.Button(channel_btn_frame, text="Edit Channel", command=self.edit_channel)
        edit_channel_btn.pack(side="left", padx=5)
        
        delete_channel_btn = ttk.Button(channel_btn_frame, text="Delete Channel", command=self.delete_channel)
        delete_channel_btn.pack(side="left", padx=5)
        
        # === Service Configuration Tab ===
        services_frame = ttk.Frame(notebook)
        notebook.add(services_frame, text="Service Configuration")
        
        # Primary service
        primary_frame = ttk.LabelFrame(services_frame, text="Primary Service", padding="5")
        primary_frame.pack(fill="x", padx=5, pady=5)
        
        self.primary_service_label = ttk.Label(primary_frame, text="Not configured")
        self.primary_service_label.pack(side="left", padx=5)
        
        edit_primary_btn = ttk.Button(primary_frame, text="Edit", command=self.edit_primary_service)
        edit_primary_btn.pack(side="right", padx=5)
        
        # Fallback services
        fallback_frame = ttk.LabelFrame(services_frame, text="Fallback Services", padding="5")
        fallback_frame.pack(fill="x", padx=5, pady=5)
        
        # Fallback toggle
        self.enable_fallback_var = tk.BooleanVar()
        fallback_check = ttk.Checkbutton(fallback_frame, text="Enable Remote Fallbacks", variable=self.enable_fallback_var, 
                                         command=self.toggle_fallback)
        fallback_check.pack(anchor="w", padx=5, pady=5)
        
        # Fallback service list
        self.fallback_container = ttk.Frame(fallback_frame)
        self.fallback_container.pack(fill="x", padx=5)
        
        # Add fallback service button
        add_fallback_btn = ttk.Button(fallback_frame, text="Add Fallback Service", command=self.add_fallback_service)
        add_fallback_btn.pack(anchor="w", padx=5, pady=5)
        
        # Ollama settings
        ollama_frame = ttk.LabelFrame(services_frame, text="Ollama Local Fallback", padding="5")
        ollama_frame.pack(fill="x", padx=5, pady=5)
        
        self.enable_ollama_var = tk.BooleanVar()
        ollama_check = ttk.Checkbutton(ollama_frame, text="Enable Ollama Fallback", variable=self.enable_ollama_var)
        ollama_check.pack(anchor="w", padx=5, pady=5)
        
        # === Control Tab ===
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Control")
        
        # Bot status
        status_frame = ttk.LabelFrame(control_frame, text="Bot Status", padding="5")
        status_frame.pack(fill="x", padx=5, pady=5)
        
        status_indicator_frame = ttk.Frame(status_frame)
        status_indicator_frame.pack(anchor="w", padx=5, pady=5)
        
        self.status_indicator = tk.Canvas(status_indicator_frame, width=20, height=20)
        self.status_indicator.pack(side="left", padx=5)
        self.update_status_indicator()
        
        self.status_label = ttk.Label(status_indicator_frame, text=BotStatus.STOPPED.value)
        self.status_label.pack(side="left", padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="Start Bot", command=self.start_bot)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Bot", command=self.stop_bot, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        save_btn = ttk.Button(button_frame, text="Save Configuration", command=self.save_config)
        save_btn.pack(side="left", padx=5)
        
        # === Logs Tab ===
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Log controls
        log_control_frame = ttk.Frame(logs_frame)
        log_control_frame.pack(fill="x", padx=5, pady=5)
        
        clear_log_btn = ttk.Button(log_control_frame, text="Clear Logs", command=lambda: self.log_text.delete(1.0, 'end'))
        clear_log_btn.pack(side="left", padx=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=20)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def load_config(self):
        """Load configuration file"""
        config_path = os.path.abspath('config.yaml')
        self.log_text.insert('end', f"Looking for config file at: {config_path}\n")
        self.log_text.insert('end', f"Current working directory: {os.getcwd()}\n")
        
        try:
            # Check if file exists
            if not os.path.exists(config_path):
                self.log_text.insert('end', "File does not exist in current directory\n")
                # Check if file exists in script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                alternative_path = os.path.join(script_dir, 'config.yaml')
                self.log_text.insert('end', f"Checking alternative path: {alternative_path}\n")
                
                if os.path.exists(alternative_path):
                    config_path = alternative_path
                    self.log_text.insert('end', "Found config file in script directory\n")
                else:
                    raise FileNotFoundError("Config file not found in both current and script directories")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
            
            # Load basic settings
            api_keys = self.config_data.get('api_keys', {})
            self.token_var.set(api_keys.get('discord_token', ''))
            
            settings = self.config_data.get('settings', {})
            self.tone_var.set(settings.get('translation_tone', ''))
            self.special_var.set(settings.get('special_instructions', ''))
            self.threshold_var.set(str(settings.get('language_separation_threshold', 1200)))
            
            # Load channel configuration
            self.refresh_channel_display()
            
            # Load service configuration
            self.refresh_service_display()
            
            # Load fallback settings
            translation_services = self.config_data.get('translation_services', {})
            self.enable_fallback_var.set(translation_services.get('enable_remote_fallback', False))
            self.enable_ollama_var.set(translation_services.get('enable_ollama_fallback', False))
            
            self.log_text.insert('end', f"Configuration loaded successfully from: {config_path}\n")
            
        except FileNotFoundError:
            self.create_default_config()
            self.log_text.insert('end', "Configuration file not found, created default configuration.\n")
        except Exception as e:
            self.log_text.insert('end', f"Error loading configuration: {e}\n")
            self.log_text.insert('end', f"Full error: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def create_default_config(self):
        """Create default configuration"""
        self.config_data = {
            'api_keys': {
                'discord_token': ''
            },
            'channels': {
                'standard': {},
                'read_only': {},
                'write_only': {}
            },
            'translation_services': {
                'primary': {
                    'provider': 'openai',
                    'base_url': 'https://openrouter.ai/api/v1',
                    'model': 'google/gemma-3-27b-it',
                    'api_key': ''
                },
                'enable_remote_fallback': False,
                'remote_fallbacks': [],
                'enable_ollama_fallback': False
            },
            'settings': {
                'translation_tone': '',
                'special_instructions': '',
                'max_retries': 2,
                'retry_delay': 3,
                'language_separation_threshold': 1200
            }
        }
    
    def save_config(self):
        """Save configuration file"""
        try:
            # Update basic settings
            if 'api_keys' not in self.config_data:
                self.config_data['api_keys'] = {}
            self.config_data['api_keys']['discord_token'] = self.token_var.get()
            
            if 'settings' not in self.config_data:
                self.config_data['settings'] = {}
            self.config_data['settings']['translation_tone'] = self.tone_var.get()
            self.config_data['settings']['special_instructions'] = self.special_var.get()
            try:
                self.config_data['settings']['language_separation_threshold'] = int(self.threshold_var.get())
            except ValueError:
                self.config_data['settings']['language_separation_threshold'] = 1200
            
            # Update fallback settings
            if 'translation_services' not in self.config_data:
                self.config_data['translation_services'] = {}
            self.config_data['translation_services']['enable_remote_fallback'] = self.enable_fallback_var.get()
            self.config_data['translation_services']['enable_ollama_fallback'] = self.enable_ollama_var.get()
            
            # Write to file
            config_path = os.path.abspath('config.yaml')
            self.log_text.insert('end', f"Saving configuration to: {config_path}\n")
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, allow_unicode=True, sort_keys=False)
            
            self.log_text.insert('end', "Configuration saved successfully!\n")
            messagebox.showinfo("Success", "Configuration saved!")
            
        except Exception as e:
            self.log_text.insert('end', f"Error saving configuration: {e}\n")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def refresh_channel_display(self):
        """Refresh channel display"""
        # Clear existing content
        for item in self.channel_tree.get_children():
            self.channel_tree.delete(item)
        
        # Add channels
        channels = self.config_data.get('channels', {})
        for channel_type, langs in channels.items():
            if isinstance(langs, dict):
                for lang, channel_ids in langs.items():
                    if isinstance(channel_ids, list):
                        for channel_id in channel_ids:
                            self.channel_tree.insert('', 'end', values=(lang, channel_id, channel_type))
                    else:
                        self.channel_tree.insert('', 'end', values=(lang, channel_ids, channel_type))
    
    def add_channel(self):
        """Add channel"""
        dialog = ChannelDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            lang, channel_id, channel_type = dialog.result
            
            # Ensure configuration structure exists
            if 'channels' not in self.config_data:
                self.config_data['channels'] = {}
            if channel_type not in self.config_data['channels']:
                self.config_data['channels'][channel_type] = {}
            
            # Add channel
            channels = self.config_data['channels'][channel_type]
            if lang not in channels:
                channels[lang] = []
            elif not isinstance(channels[lang], list):
                channels[lang] = [channels[lang]]
            
            channels[lang].append(channel_id)
            
            # Refresh display
            self.refresh_channel_display()
            self.log_text.insert('end', f"Added channel: {lang} - {channel_id} ({channel_type})\n")
    
    def edit_channel(self):
        """Edit channel"""
        selected = self.channel_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a channel to edit!")
            return
        
        item = selected[0]
        values = self.channel_tree.item(item)['values']
        
        dialog = ChannelDialog(self.root, "Edit Channel", values)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            # Remove old channel
            self.delete_channel(skip_confirm=True)
            
            # Add new channel
            lang, channel_id, channel_type = dialog.result
            
            # Ensure configuration structure exists
            if 'channels' not in self.config_data:
                self.config_data['channels'] = {}
            if channel_type not in self.config_data['channels']:
                self.config_data['channels'][channel_type] = {}
            
            # Add channel
            channels = self.config_data['channels'][channel_type]
            if lang not in channels:
                channels[lang] = []
            elif not isinstance(channels[lang], list):
                channels[lang] = [channels[lang]]
            
            channels[lang].append(channel_id)
            
            # Refresh display
            self.refresh_channel_display()
    
    def delete_channel(self, skip_confirm=False):
        """Delete channel"""
        selected = self.channel_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a channel to delete!")
            return
        
        if not skip_confirm and not messagebox.askyesno("Confirm", "Are you sure you want to delete the selected channel(s)?"):
            return
        
        for item in selected:
            values = self.channel_tree.item(item)['values']
            lang, channel_id, channel_type = values
            
            # Remove from configuration
            if channel_type in self.config_data.get('channels', {}):
                channels = self.config_data['channels'][channel_type]
                if lang in channels:
                    if isinstance(channels[lang], list):
                        if channel_id in channels[lang]:
                            channels[lang].remove(channel_id)
                            if not channels[lang]:
                                del channels[lang]
                    else:
                        if channels[lang] == channel_id:
                            del channels[lang]
        
        # Refresh display
        self.refresh_channel_display()
    
    def refresh_service_display(self):
        """Refresh service display"""
        translation_services = self.config_data.get('translation_services', {})
        
        # Update primary service display
        primary = translation_services.get('primary', {})
        if primary:
            text = f"{primary.get('provider', '')} - {primary.get('model', '')}"
            self.primary_service_label.config(text=text)
        else:
            self.primary_service_label.config(text="Not configured")
        
        # Clear and rebuild fallback service list
        for widget in self.fallback_container.winfo_children():
            widget.destroy()
        
        fallbacks = translation_services.get('remote_fallbacks', [])
        for i, fallback in enumerate(fallbacks):
            frame = ttk.Frame(self.fallback_container)
            frame.pack(fill="x", pady=2)
            
            provider = fallback.get('provider', '')
            model = fallback.get('model', '')
            label = ttk.Label(frame, text=f"{i+1}. {provider} - {model}")
            label.pack(side="left", padx=5)
            
            edit_btn = ttk.Button(frame, text="Edit", 
                                  command=lambda idx=i: self.edit_fallback_service(idx))
            edit_btn.pack(side="right", padx=2)
            
            delete_btn = ttk.Button(frame, text="Delete", 
                                    command=lambda idx=i: self.delete_fallback_service(idx))
            delete_btn.pack(side="right", padx=2)
    
    def edit_primary_service(self):
        """Edit primary service"""
        translation_services = self.config_data.get('translation_services', {})
        current_primary = translation_services.get('primary', {})
        
        dialog = ServiceDialog(self.root, "Edit Primary Service", current_primary)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            if 'translation_services' not in self.config_data:
                self.config_data['translation_services'] = {}
            self.config_data['translation_services']['primary'] = dialog.result
            self.refresh_service_display()
    
    def toggle_fallback(self):
        """Toggle fallback status"""
        enabled = self.enable_fallback_var.get()
        if 'translation_services' not in self.config_data:
            self.config_data['translation_services'] = {}
        self.config_data['translation_services']['enable_remote_fallback'] = enabled
    
    def add_fallback_service(self):
        """Add fallback service"""
        dialog = ServiceDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            if 'translation_services' not in self.config_data:
                self.config_data['translation_services'] = {}
            if 'remote_fallbacks' not in self.config_data['translation_services']:
                self.config_data['translation_services']['remote_fallbacks'] = []
            
            self.config_data['translation_services']['remote_fallbacks'].append(dialog.result)
            self.refresh_service_display()
    
    def edit_fallback_service(self, index):
        """Edit fallback service"""
        fallbacks = self.config_data.get('translation_services', {}).get('remote_fallbacks', [])
        if 0 <= index < len(fallbacks):
            dialog = ServiceDialog(self.root, "Edit Fallback Service", fallbacks[index])
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                fallbacks[index] = dialog.result
                self.refresh_service_display()
    
    def delete_fallback_service(self, index):
        """Delete fallback service"""
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this fallback service?"):
            fallbacks = self.config_data.get('translation_services', {}).get('remote_fallbacks', [])
            if 0 <= index < len(fallbacks):
                del fallbacks[index]
                self.refresh_service_display()
    
    def start_bot(self):
        """Start the bot"""
        if self.bot_process is not None:
            self.log_text.insert('end', "Bot is already running!\n")
            return
        
        # Save configuration
        self.save_config()
        
        try:
            # Start subprocess
            self.bot_process = subprocess.Popen(
                ['python', 'bot.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Create thread to read output
            threading.Thread(target=self.read_bot_output, daemon=True).start()
            
            # Update UI status
            self.current_status = BotStatus.RUNNING
            self.update_status_indicator()
            self.status_label.config(text=BotStatus.RUNNING.value)
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            self.log_text.insert('end', "Bot started successfully!\n")
            
        except Exception as e:
            self.log_text.insert('end', f"Error starting bot: {e}\n")
            messagebox.showerror("Error", f"Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the bot"""
        if self.bot_process is None:
            self.log_text.insert('end', "Bot is not running!\n")
            return
        
        try:
            # Terminate subprocess
            self.bot_process.terminate()
            self.bot_process.wait(timeout=5)
            
        except subprocess.TimeoutExpired:
            self.bot_process.kill()
            self.log_text.insert('end', "Force killed bot.\n")
        
        finally:
            self.bot_process = None
            
            # Update UI status
            self.current_status = BotStatus.STOPPED
            self.update_status_indicator()
            self.status_label.config(text=BotStatus.STOPPED.value)
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
            self.log_text.insert('end', "Bot stopped.\n")
    
    def read_bot_output(self):
        """Read bot output"""
        while self.bot_process and self.bot_process.poll() is None:
            try:
                line = self.bot_process.stdout.readline()
                if line:
                    self.log_queue.put(line.strip())
            except Exception as e:
                self.log_queue.put(f"Output reading error: {e}")
                break
    
    def check_log_queue(self):
        """Check log queue and update display"""
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.insert('end', line + '\n')
                self.log_text.see('end')
                
                # Check error status
                if 'error' in line.lower() or 'critical' in line.lower():
                    self.current_status = BotStatus.ERROR
                    self.update_status_indicator()
                
        except queue.Empty:
            pass
        
        # Check periodically
        self.root.after(100, self.check_log_queue)
    
    def update_status_indicator(self):
        """Update status indicator"""
        self.status_indicator.delete("all")
        
        colors = {
            BotStatus.STOPPED: "red",
            BotStatus.RUNNING: "green",
            BotStatus.ERROR: "orange"
        }
        
        color = colors.get(self.current_status, "gray")
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline="black")
    
    def on_closing(self):
        """Handle window closing"""
        if self.bot_process is not None:
            if messagebox.askyesno("Confirm", "Bot is running, are you sure you want to exit?"):
                self.stop_bot()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run main loop"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TranslationBotUI()
    app.run()