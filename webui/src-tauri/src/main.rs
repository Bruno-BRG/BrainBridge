#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use tauri_plugin_shell::ShellExt;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Backend Python como sidecar (mesma API da Fase 1).
            // Em dev: script que usa o venv do repo. No bundle: binario PyInstaller.
            let (_rx, child) = app
                .shell()
                .sidecar("brainbridge-server")?
                .args(["--port", "8000"])
                .spawn()?;
            // Guarda o handle no estado: o Tauri encerra sidecars ao sair.
            app.manage(child);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
