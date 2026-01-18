/**
 * HuggingFace Downloader Extension
 * Downloads and auto-merges split safetensor files from HuggingFace repos
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[HF Downloader] Loading extension...");

// Add menu button to ComfyUI
function addMenuButton() {
    const buttonGroup = document.querySelector(".comfyui-button-group");

    if (!buttonGroup) {
        console.warn("[HF Downloader] Button group not found, retrying...");
        setTimeout(addMenuButton, 500);
        return;
    }

    if (document.getElementById("hf-downloader-button")) {
        console.log("[HF Downloader] Button already exists");
        return;
    }

    const hfButton = document.createElement("button");
    hfButton.textContent = "HF Downloader";
    hfButton.id = "hf-downloader-button";
    hfButton.title = "Download HuggingFace Models";

    hfButton.onclick = async () => {
        if (!window.hfDownloaderUI) {
            console.log("[HF Downloader] Creating UI instance...");
            window.hfDownloaderUI = new HFDownloaderUI();
            document.body.appendChild(window.hfDownloaderUI.modal);
        }

        if (window.hfDownloaderUI) {
            window.hfDownloaderUI.openModal();
        }
    };

    // Create wrapper div (like ComfyUI Manager does)
    const buttonGroupWrapper = document.createElement("div");
    buttonGroupWrapper.className = "comfyui-button-group";
    buttonGroupWrapper.appendChild(hfButton);

    // Append to parent container
    const parent = buttonGroup.parentElement;
    parent.appendChild(buttonGroupWrapper);
    console.log("[HF Downloader] Button group added to parent container");
}

// Main UI class
class HFDownloaderUI {
    constructor() {
        this.modal = null;
        this.currentTasks = new Map();
        this.buildModal();
    }

    buildModal() {
        this.modal = document.createElement("div");
        this.modal.className = "hf-downloader-modal";
        this.modal.id = "hf-downloader-modal";

        this.modal.innerHTML = `
            <div class="hf-downloader-modal-content">
                <div class="hf-downloader-header">
                    <h2>HuggingFace Model Downloader</h2>
                    <button class="hf-close-button" id="hf-close-modal">×</button>
                </div>

                <div class="hf-input-section">
                    <input type="text" id="hf-repo-input" placeholder="Enter HuggingFace repo (e.g., username/model-name)" />
                    <button id="hf-scan-button">Scan Repo</button>
                </div>

                <div class="hf-models-section" id="hf-models-section" style="display: none;">
                    <h3>Detected Models:</h3>
                    <table class="hf-models-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Type</th>
                                <th>Files</th>
                                <th>Output Name</th>
                                <th>Destination</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="hf-models-tbody"></tbody>
                    </table>
                </div>

                <div class="hf-progress-section" id="hf-progress-section"></div>
            </div>
        `;

        // Cache DOM elements
        this.closeButton = this.modal.querySelector("#hf-close-modal");
        this.repoInput = this.modal.querySelector("#hf-repo-input");
        this.scanButton = this.modal.querySelector("#hf-scan-button");
        this.modelsSection = this.modal.querySelector("#hf-models-section");
        this.modelsTbody = this.modal.querySelector("#hf-models-tbody");
        this.progressSection = this.modal.querySelector("#hf-progress-section");

        // Setup event listeners
        this.closeButton.onclick = () => this.closeModal();
        this.scanButton.onclick = () => this.scanRepo();
        this.repoInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") this.scanRepo();
        });

        // Close on background click
        this.modal.addEventListener("click", (e) => {
            if (e.target === this.modal) this.closeModal();
        });
    }

    openModal() {
        this.modal.classList.add("open");
    }

    closeModal() {
        this.modal.classList.remove("open");
    }

    async scanRepo() {
        const repoId = this.repoInput.value.trim();

        if (!repoId || !repoId.includes("/")) {
            alert("Please enter a valid repo ID (username/model-name)");
            return;
        }

        this.scanButton.disabled = true;
        this.scanButton.textContent = "Scanning...";
        this.modelsSection.style.display = "none";

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

            this.renderModels(repoId, data.models);

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            this.scanButton.disabled = false;
            this.scanButton.textContent = "Scan Repo";
        }
    }

    renderModels(repoId, models) {
        this.modelsTbody.innerHTML = "";

        if (models.length === 0) {
            this.modelsTbody.innerHTML = '<tr><td colspan="6">No safetensor files found</td></tr>';
            this.modelsSection.style.display = "block";
            return;
        }

        models.forEach((model, index) => {
            const row = document.createElement("tr");

            const typeBadge = model.is_split ?
                '<span class="badge badge-split">Split Files</span>' :
                '<span class="badge badge-single">Single File</span>';

            row.innerHTML = `
                <td>${model.path}</td>
                <td>${typeBadge}</td>
                <td>${model.file_count}</td>
                <td><input type="text" class="output-name-input" value="${model.suggested_name}" id="name-${index}" /></td>
                <td>
                    <select class="model-type-select" id="type-${index}">
                        <option value="checkpoint">Checkpoint</option>
                        <option value="lora">LoRA</option>
                        <option value="vae">VAE</option>
                        <option value="upscale_model">Upscale</option>
                        <option value="embedding">Embedding</option>
                        <option value="clip">CLIP</option>
                        <option value="controlnet">ControlNet</option>
                        <option value="diffusion_model">Diffusion Models</option>
                        <option value="text_encoder">Text Encoder</option>
                    </select>
                </td>
                <td><button class="download-btn" data-index="${index}">Download</button></td>
            `;

            const downloadBtn = row.querySelector(".download-btn");
            downloadBtn.onclick = () => this.downloadModel(repoId, model, index);

            this.modelsTbody.appendChild(row);
        });

        this.modelsSection.style.display = "block";
    }

    async downloadModel(repoId, model, index) {
        const outputName = document.getElementById(`name-${index}`).value.trim();
        const modelType = document.getElementById(`type-${index}`).value;

        if (!outputName) {
            alert("Output name cannot be empty");
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

            this.trackProgress(data.task_id, outputName);

        } catch (error) {
            alert(`Download failed: ${error.message}`);
        }
    }

    async trackProgress(taskId, modelName) {
        const progressDiv = document.createElement("div");
        progressDiv.className = "progress-item";
        progressDiv.id = `progress-${taskId}`;
        progressDiv.innerHTML = `
            <div class="progress-header">
                <span class="progress-name">${modelName}</span>
                <span class="progress-stage" id="stage-${taskId}">Starting...</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="bar-${taskId}"></div>
            </div>
            <div class="progress-message" id="msg-${taskId}"></div>
            <div class="progress-actions" id="actions-${taskId}" style="display: none; margin-top: 10px;"></div>
        `;

        this.progressSection.appendChild(progressDiv);
        this.currentTasks.set(taskId, { div: progressDiv, modelName });

        const pollInterval = setInterval(async () => {
            try {
                const response = await api.fetchApi(`/hf_downloader/progress/${taskId}`);
                const progress = await response.json();

                this.updateProgress(taskId, progress);

                if (progress.status === "completed" || progress.status === "error") {
                    clearInterval(pollInterval);
                    this.addProgressActions(taskId, progress.status, modelName);
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
            // Make ERROR text red
            if (progress.status === "error") {
                stageEl.style.color = "#ff6b6b";
            }
        }
        if (msgEl) msgEl.textContent = progress.message;

        if (barEl && progress.total > 0) {
            const percent = (progress.current / progress.total) * 100;
            barEl.style.width = `${percent}%`;

            if (progress.status === "completed") {
                barEl.classList.add("completed");
            } else if (progress.status === "error") {
                barEl.classList.add("error");
            }
        }
    }

    addProgressActions(taskId, status, modelName) {
        const actionsDiv = document.getElementById(`actions-${taskId}`);
        if (!actionsDiv) return;

        // Clear existing buttons to prevent duplicates
        actionsDiv.innerHTML = "";
        actionsDiv.style.display = "flex";
        actionsDiv.style.gap = "10px";

        // Always add clear button
        const clearBtn = document.createElement("button");
        clearBtn.textContent = "Clear";
        clearBtn.className = "action-btn clear-btn";
        clearBtn.onclick = () => this.clearProgress(taskId);
        actionsDiv.appendChild(clearBtn);

        // Add retry button only on error
        if (status === "error") {
            const taskData = this.currentTasks.get(taskId);
            const retryBtn = document.createElement("button");
            retryBtn.textContent = "Retry";
            retryBtn.className = "action-btn retry-btn";
            retryBtn.onclick = () => {
                alert("Retry functionality: Please use the download button above to retry the download.");
            };
            actionsDiv.insertBefore(retryBtn, clearBtn);
        }
    }

    clearProgress(taskId) {
        const progressDiv = document.getElementById(`progress-${taskId}`);
        if (progressDiv) {
            progressDiv.remove();
        }
        this.currentTasks.delete(taskId);
    }
}

// Add CSS styles
function addStyles() {
    const style = document.createElement("style");
    style.textContent = `
        .hf-downloader-modal {
            position: fixed;
            z-index: 1001;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0s linear 0.3s;
        }

        .hf-downloader-modal.open {
            opacity: 1;
            visibility: visible;
            transition: opacity 0.3s ease, visibility 0s linear 0s;
        }

        .hf-downloader-modal-content {
            background: var(--comfy-menu-bg, #1a1a1a);
            color: var(--comfy-text-color, #fff);
            border-radius: 8px;
            padding: 20px;
            max-width: 900px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        }

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
        }

        .hf-close-button {
            background: none;
            border: none;
            color: #fff;
            font-size: 30px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
        }

        .hf-close-button:hover {
            color: #ff6b6b;
        }

        .hf-input-section {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        #hf-repo-input {
            flex: 1;
            padding: 10px;
            background: #2a2a2a;
            border: 1px solid #444;
            color: #fff;
            border-radius: 4px;
        }

        #hf-scan-button {
            padding: 10px 20px;
            background: #4a9eff;
            border: none;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }

        #hf-scan-button:hover {
            background: #3a8eef;
        }

        #hf-scan-button:disabled {
            background: #666;
            cursor: not-allowed;
        }

        .hf-models-table {
            width: 100%;
            border-collapse: collapse;
            background: #2a2a2a;
        }

        .hf-models-table th,
        .hf-models-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }

        .hf-models-table th {
            background: #333;
            font-weight: bold;
        }

        .badge {
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

        .output-name-input,
        .model-type-select {
            padding: 6px;
            background: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            border-radius: 3px;
            width: 100%;
        }

        .download-btn {
            padding: 6px 12px;
            background: #4a9eff;
            border: none;
            color: #fff;
            border-radius: 3px;
            cursor: pointer;
        }

        .download-btn:hover {
            background: #3a8eef;
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
            width: 0%;
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

        .action-btn {
            padding: 6px 12px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
        }

        .retry-btn {
            background: #4a9eff;
            color: #fff;
        }

        .retry-btn:hover {
            background: #3a8eef;
        }

        .clear-btn {
            background: #666;
            color: #fff;
        }

        .clear-btn:hover {
            background: #555;
        }
    `;
    document.head.appendChild(style);
}

// Register extension with ComfyUI
app.registerExtension({
    name: "HFDownloader",
    async setup() {
        console.log("[HF Downloader] Setting up extension...");
        addStyles();
        addMenuButton();
        console.log("[HF Downloader] Extension setup complete");
    }
});
