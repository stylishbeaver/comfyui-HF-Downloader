/**
 * HuggingFace Downloader Frontend
 * Downloads and auto-merges split safetensor files from HuggingFace repos
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyDialog } from "../../../scripts/ui.js";

class HFDownloaderDialog extends ComfyDialog {
    constructor() {
        super();
        this.element = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", [
                $el("div.hf-downloader-header", [
                    $el("h2", {}, ["HuggingFace Model Downloader"]),
                    $el("button.close-button", {
                        textContent: "×",
                        onclick: () => this.close()
                    })
                ]),
                this.createRepoInput(),
                this.createModelsContainer(),
                this.createProgressContainer()
            ])
        ]);

        this.models = [];
        this.currentTasks = new Map();
    }

    createRepoInput() {
        this.repoInput = $el("input.hf-repo-input", {
            type: "text",
            placeholder: "Enter HuggingFace repo (e.g., username/model-name)",
            onkeypress: (e) => {
                if (e.key === "Enter") {
                    this.scanRepo();
                }
            }
        });

        this.scanButton = $el("button.hf-scan-button", {
            textContent: "Scan Repo",
            onclick: () => this.scanRepo()
        });

        return $el("div.hf-input-container", [
            this.repoInput,
            this.scanButton
        ]);
    }

    createModelsContainer() {
        this.modelsContainer = $el("div.hf-models-container", {
            style: { display: "none" }
        });

        this.modelsTable = $el("table.hf-models-table", [
            $el("thead", [
                $el("tr", [
                    $el("th", {}, ["Model"]),
                    $el("th", {}, ["Type"]),
                    $el("th", {}, ["Files"]),
                    $el("th", {}, ["Output Name"]),
                    $el("th", {}, ["Destination"]),
                    $el("th", {}, ["Action"])
                ])
            ]),
            $el("tbody")
        ]);

        this.modelsContainer.appendChild($el("h3", {}, ["Detected Models:"]));
        this.modelsContainer.appendChild(this.modelsTable);

        return this.modelsContainer;
    }

    createProgressContainer() {
        this.progressContainer = $el("div.hf-progress-container", {
            style: { display: "none" }
        });

        return this.progressContainer;
    }

    async scanRepo() {
        const repoId = this.repoInput.value.trim();

        if (!repoId) {
            this.showError("Please enter a repository ID");
            return;
        }

        if (!repoId.includes('/')) {
            this.showError("Invalid format. Use: username/model-name");
            return;
        }

        this.scanButton.disabled = true;
        this.scanButton.textContent = "Scanning...";
        this.modelsContainer.style.display = "none";

        try {
            const response = await api.fetchApi("/hf_downloader/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_id: repoId })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to scan repository");
            }

            this.models = data.models;
            this.renderModels(repoId);

        } catch (error) {
            this.showError(`Error: ${error.message}`);
        } finally {
            this.scanButton.disabled = false;
            this.scanButton.textContent = "Scan Repo";
        }
    }

    renderModels(repoId) {
        const tbody = this.modelsTable.querySelector("tbody");
        tbody.innerHTML = "";

        if (this.models.length === 0) {
            tbody.appendChild($el("tr", [
                $el("td", { colspan: 6, textContent: "No safetensor files found in repository" })
            ]));
            this.modelsContainer.style.display = "block";
            return;
        }

        this.models.forEach((model, index) => {
            const row = this.createModelRow(repoId, model, index);
            tbody.appendChild(row);
        });

        this.modelsContainer.style.display = "block";
    }

    createModelRow(repoId, model, index) {
        // Model type badge
        const typeText = model.is_split ? "Split Files" : "Single File";
        const typeBadge = $el("span.model-type-badge", {
            className: model.is_split ? "badge-split" : "badge-single",
            textContent: typeText
        });

        // Output name input
        const nameInput = $el("input.output-name-input", {
            type: "text",
            value: model.suggested_name,
            id: `output-name-${index}`
        });

        // Model type selector
        const typeSelect = $el("select.model-type-select", {
            id: `model-type-${index}`
        }, [
            $el("option", { value: "checkpoint", textContent: "Checkpoint" }),
            $el("option", { value: "lora", textContent: "LoRA" }),
            $el("option", { value: "vae", textContent: "VAE" }),
            $el("option", { value: "upscale_model", textContent: "Upscale" }),
            $el("option", { value: "embedding", textContent: "Embedding" }),
            $el("option", { value: "clip", textContent: "CLIP" }),
            $el("option", { value: "controlnet", textContent: "ControlNet" }),
            $el("option", { value: "diffusion_model", textContent: "Diffusion" }),
            $el("option", { value: "text_encoder", textContent: "Text Encoder" })
        ]);

        // Download button
        const downloadBtn = $el("button.hf-download-btn", {
            textContent: "Download",
            onclick: () => this.downloadModel(repoId, model, index)
        });

        const row = $el("tr", [
            $el("td", { textContent: model.path }),
            $el("td", [typeBadge]),
            $el("td", { textContent: model.file_count }),
            $el("td", [nameInput]),
            $el("td", [typeSelect]),
            $el("td", [downloadBtn])
        ]);

        return row;
    }

    async downloadModel(repoId, model, index) {
        const outputName = document.getElementById(`output-name-${index}`).value.trim();
        const modelType = document.getElementById(`model-type-${index}`).value;

        if (!outputName) {
            this.showError("Output name cannot be empty");
            return;
        }

        try {
            const response = await api.fetchApi("/hf_downloader/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    repo_id: repoId,
                    model_path: model.path,
                    files: model.files,
                    output_name: outputName,
                    model_type: modelType
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to start download");
            }

            // Start progress tracking
            this.trackProgress(data.task_id, outputName);

        } catch (error) {
            this.showError(`Download failed: ${error.message}`);
        }
    }

    async trackProgress(taskId, modelName) {
        const progressDiv = $el("div.progress-item", {
            id: `progress-${taskId}`
        }, [
            $el("div.progress-header", [
                $el("span.progress-name", { textContent: modelName }),
                $el("span.progress-stage", { id: `stage-${taskId}`, textContent: "Starting..." })
            ]),
            $el("div.progress-bar-container", [
                $el("div.progress-bar", { id: `bar-${taskId}`, style: { width: "0%" } })
            ]),
            $el("div.progress-message", { id: `msg-${taskId}`, textContent: "" })
        ]);

        this.progressContainer.appendChild(progressDiv);
        this.progressContainer.style.display = "block";
        this.currentTasks.set(taskId, { div: progressDiv, modelName });

        // Poll for progress
        const pollInterval = setInterval(async () => {
            try {
                const response = await api.fetchApi(`/hf_downloader/progress/${taskId}`);
                const progress = await response.json();

                this.updateProgress(taskId, progress);

                if (progress.status === "completed" || progress.status === "error") {
                    clearInterval(pollInterval);

                    if (progress.status === "completed") {
                        this.showSuccess(`Download completed: ${modelName}`);
                    } else {
                        this.showError(`Download failed: ${progress.message}`);
                    }
                }
            } catch (error) {
                console.error("Error polling progress:", error);
                clearInterval(pollInterval);
            }
        }, 500);
    }

    updateProgress(taskId, progress) {
        const stageEl = document.getElementById(`stage-${taskId}`);
        const barEl = document.getElementById(`bar-${taskId}`);
        const msgEl = document.getElementById(`msg-${taskId}`);

        if (stageEl) {
            stageEl.textContent = progress.stage.toUpperCase();
        }

        if (barEl && progress.total > 0) {
            const percent = (progress.current / progress.total) * 100;
            barEl.style.width = `${percent}%`;
        }

        if (msgEl) {
            msgEl.textContent = progress.message;
        }

        // Add completion class
        if (progress.status === "completed") {
            barEl.classList.add("completed");
        } else if (progress.status === "error") {
            barEl.classList.add("error");
        }
    }

    showError(message) {
        app.ui.dialog.show($el("div", [
            $el("p", { textContent: message, style: { color: "#ff6b6b" } })
        ]));
    }

    showSuccess(message) {
        app.ui.dialog.show($el("div", [
            $el("p", { textContent: message, style: { color: "#51cf66" } })
        ]));
    }

    show() {
        this.element.style.display = "flex";
    }

    close() {
        this.element.style.display = "none";
    }
}

// Utility function to create DOM elements
function $el(tag, attrs = {}, children = []) {
    const element = document.createElement(tag.split(".")[0]);

    // Handle classes
    if (tag.includes(".")) {
        element.className = tag.split(".").slice(1).join(" ");
    }

    // Set attributes
    for (const [key, value] of Object.entries(attrs)) {
        if (key === "style" && typeof value === "object") {
            Object.assign(element.style, value);
        } else if (key.startsWith("on")) {
            element.addEventListener(key.substring(2), value);
        } else if (key === "textContent") {
            element.textContent = value;
        } else if (key === "className") {
            element.className = value;
        } else {
            element.setAttribute(key, value);
        }
    }

    // Append children
    for (const child of children) {
        if (typeof child === "string") {
            element.appendChild(document.createTextNode(child));
        } else {
            element.appendChild(child);
        }
    }

    return element;
}

// Register extension with ComfyUI
app.registerExtension({
    name: "HFDownloader",

    async setup() {
        // Add styles first
        const style = document.createElement("style");
        style.textContent = `
            .hf-downloader-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 2px solid #444;
                padding-bottom: 10px;
            }

            .hf-downloader-header h2 {
                margin: 0;
                color: #fff;
            }

            .hf-downloader-header .close-button {
                background: none;
                border: none;
                color: #fff;
                font-size: 30px;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
                line-height: 30px;
            }

            .hf-downloader-header .close-button:hover {
                color: #ff6b6b;
            }

            .hf-input-container {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }

            .hf-repo-input {
                flex: 1;
                padding: 10px;
                background: #2a2a2a;
                border: 1px solid #444;
                color: #fff;
                border-radius: 4px;
            }

            .hf-scan-button, .hf-download-btn {
                padding: 10px 20px;
                background: #4a9eff;
                border: none;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
            }

            .hf-scan-button:hover, .hf-download-btn:hover {
                background: #3a8eef;
            }

            .hf-scan-button:disabled {
                background: #666;
                cursor: not-allowed;
            }

            .hf-models-container {
                margin-bottom: 20px;
            }

            .hf-models-container h3 {
                color: #fff;
                margin-bottom: 10px;
            }

            .hf-models-table {
                width: 100%;
                border-collapse: collapse;
                background: #2a2a2a;
            }

            .hf-models-table th, .hf-models-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #444;
                color: #fff;
            }

            .hf-models-table th {
                background: #333;
                font-weight: bold;
            }

            .model-type-badge {
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }

            .badge-split {
                background: #ff922b;
                color: #000;
            }

            .badge-single {
                background: #51cf66;
                color: #000;
            }

            .output-name-input, .model-type-select {
                padding: 6px;
                background: #1a1a1a;
                border: 1px solid #444;
                color: #fff;
                border-radius: 3px;
                width: 100%;
            }

            .hf-progress-container {
                margin-top: 20px;
            }

            .progress-item {
                margin-bottom: 15px;
                padding: 15px;
                background: #2a2a2a;
                border-radius: 4px;
            }

            .progress-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }

            .progress-name {
                font-weight: bold;
                color: #fff;
            }

            .progress-stage {
                color: #4a9eff;
                font-size: 12px;
            }

            .progress-bar-container {
                width: 100%;
                height: 20px;
                background: #1a1a1a;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 8px;
            }

            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #4a9eff 0%, #3a8eef 100%);
                transition: width 0.3s ease;
            }

            .progress-bar.completed {
                background: linear-gradient(90deg, #51cf66 0%, #40bf56 100%);
            }

            .progress-bar.error {
                background: linear-gradient(90deg, #ff6b6b 0%, #ef5b5b 100%);
            }

            .progress-message {
                font-size: 13px;
                color: #aaa;
            }
        `;
        document.head.appendChild(style);

        // Add menu button with retry logic
        const addMenuButton = () => {
            // Try multiple selectors to find the right menu location
            const buttonGroup = document.querySelector(".comfyui-button-group");
            const menu = document.querySelector(".comfy-menu");
            const targetElement = buttonGroup || menu;

            if (!targetElement) {
                console.warn("[HF Downloader] Menu not found, retrying...");
                setTimeout(addMenuButton, 500);
                return;
            }

            const hfButton = document.createElement("button");
            hfButton.textContent = "HF Downloader";
            hfButton.style.cssText = "margin: 4px; padding: 4px 8px; background: #4a9eff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;";
            hfButton.onclick = () => {
                if (!this.dialog) {
                    this.dialog = new HFDownloaderDialog();
                }
                this.dialog.show();
            };

            targetElement.appendChild(hfButton);
            console.log("[HF Downloader] Button added to:", targetElement.className);
        };

        addMenuButton();
    }
});
