#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use tauri::Manager;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Backend Python como sidecar (mesma API da Fase 1).
            // Em dev: script que usa o venv do repo. No bundle: binario PyInstaller.
            // O launcher Windows do repositorio chama o Python 3.12 instalado na
            // maquina; disponibiliza as dependencias do venv para esse Python.
            let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..");
            let venv_site_packages = repo_root.join(".venv").join("Lib").join("site-packages");

            let mut sidecar = app
                .shell()
                .sidecar("brainbridge-server")?
                .args(["--port", "8000"])
                .current_dir(&repo_root);
            if venv_site_packages.is_dir() {
                sidecar = sidecar.env("PYTHONPATH", venv_site_packages.as_os_str());
            }
            let (mut rx, child) = sidecar.spawn()?;
            // Drena stdout/stderr do sidecar: sem leitor o pipe fecha e
            // qualquer `print()` no backend Python estoura a requisicao
            // (era a causa dos 400/500 no /devices/unity/*). Repassa os
            // logs para o console do `tauri dev`.
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            print!("[sidecar] {}", String::from_utf8_lossy(&line))
                        }
                        CommandEvent::Stderr(line) => {
                            eprint!("[sidecar] {}", String::from_utf8_lossy(&line))
                        }
                        _ => {}
                    }
                }
            });
            // Guarda o handle no estado: o Tauri encerra sidecars ao sair.
            app.manage(child);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
