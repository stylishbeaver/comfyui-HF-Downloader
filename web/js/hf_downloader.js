/**
 * HuggingFace Downloader Extension
 * Downloads and auto-merges split safetensor files and single GGUF files from HuggingFace repos
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

                <div class="hf-tabs">
                    <button class="hf-tab active" data-tab="download">Download</button>
                    <button class="hf-tab" data-tab="status">Status <span id="hf-status-badge" class="hf-status-badge" style="display:none">0</span></button>
                    <button class="hf-tab" data-tab="manage">Manage Files</button>
                </div>

                <div id="hf-tab-download" class="hf-tab-content active">
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
                                    <th>Size</th>
                                    <th>Precision</th>
                                    <th>Quant</th>
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

                <div id="hf-tab-status" class="hf-tab-content">
                    <div class="hf-status-toolbar">
                        <button id="hf-status-refresh-btn">↻ Refresh</button>
                        <button id="hf-status-clear-done-btn">Clear Done</button>
                    </div>
                    <div id="hf-status-active-section">
                        <div class="hf-status-section-label">Active &amp; Queued</div>
                        <div id="hf-status-active-list"></div>
                    </div>
                    <div id="hf-status-done-section">
                        <div class="hf-status-section-label">Completed</div>
                        <div id="hf-status-done-list"></div>
                    </div>
                    <div id="hf-status-failed-section">
                        <div class="hf-status-section-label">Failed</div>
                        <div id="hf-status-failed-list"></div>
                    </div>
                </div>

                <div id="hf-tab-manage" class="hf-tab-content">
                    <div class="hf-manage-controls">
                        <label>Filter by type:</label>
                        <select id="hf-file-type-filter">
                            <option value="checkpoint">Checkpoints</option>
                            <option value="lora">LoRAs</option>
                            <option value="vae">VAE</option>
                            <option value="diffusion_model">Diffusion Models</option>
                            <option value="text_encoder">Text Encoders</option>
                            <option value="upscale_model">Upscale Models</option>
                            <option value="controlnet">ControlNet</option>
                            <option value="embedding">Embeddings</option>
                            <option value="clip">CLIP</option>
                        </select>
                        <button id="hf-refresh-files">Refresh</button>
                    </div>
                    <div id="hf-files-list"></div>
                </div>
            </div>
        `;

        // Cache DOM elements
        this.closeButton = this.modal.querySelector("#hf-close-modal");
        this.repoInput = this.modal.querySelector("#hf-repo-input");
        this.scanButton = this.modal.querySelector("#hf-scan-button");
        this.modelsSection = this.modal.querySelector("#hf-models-section");
        this.modelsTbody = this.modal.querySelector("#hf-models-tbody");
        this.progressSection = this.modal.querySelector("#hf-progress-section");
        this.fileTypeFilter = this.modal.querySelector("#hf-file-type-filter");
        this.refreshFilesButton = this.modal.querySelector("#hf-refresh-files");
        this.filesList = this.modal.querySelector("#hf-files-list");
        this.statusBadge = this.modal.querySelector("#hf-status-badge");
        this.statusActiveList = this.modal.querySelector("#hf-status-active-list");
        this.statusDoneList = this.modal.querySelector("#hf-status-done-list");
        this.statusFailedList = this.modal.querySelector("#hf-status-failed-list");
        this.statusActiveSection = this.modal.querySelector("#hf-status-active-section");
        this.statusDoneSection = this.modal.querySelector("#hf-status-done-section");
        this.statusFailedSection = this.modal.querySelector("#hf-status-failed-section");
        this.statusInterval = null;

        // Setup event listeners
        this.closeButton.onclick = () => this.closeModal();
        this.scanButton.onclick = () => this.scanRepo();
        this.refreshFilesButton.onclick = () => this.loadFiles();
        this.fileTypeFilter.onchange = () => this.loadFiles();
        this.modal.querySelector("#hf-status-refresh-btn").onclick = () => this.refreshStatus();
        this.modal.querySelector("#hf-status-clear-done-btn").onclick = () => this.clearDoneItems();

        // Tab switching
        this.modal.querySelectorAll(".hf-tab").forEach(tab => {
            tab.onclick = () => this.switchTab(tab.dataset.tab);
        });

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

    formatSize(bytes) {
        if (!bytes) return 'N/A';
        const gb = bytes / (1024 * 1024 * 1024);
        if (gb >= 1) {
            return `${gb.toFixed(2)} GB`;
        }
        const mb = bytes / (1024 * 1024);
        return `${mb.toFixed(2)} MB`;
    }

    inferModelTypeFromPath(modelPath) {
        if (!modelPath || modelPath === "root") {
            return null;
        }

        const normalized = modelPath.toLowerCase().replace(/\\/g, "/");
        const segments = normalized.split("/").filter(Boolean);

        const patterns = [
            { re: /^(checkpoints?|ckpt)$/, type: "checkpoint" },
            { re: /^loras?$/, type: "lora" },
            { re: /^vaes?$/, type: "vae" },
            { re: /^(upscale_models?|upscalers?|upscale)$/, type: "upscale_model" },
            { re: /^embeddings?$/, type: "embedding" },
            { re: /^clip$/, type: "clip" },
            { re: /^controlnets?$/, type: "controlnet" },
            { re: /^(diffusion_models?|diffusion_model|unet)$/, type: "diffusion_model" },
            { re: /^text[_-]?encoders?$/, type: "text_encoder" },
            { re: /^text[_-]?encoder$/, type: "text_encoder" }
        ];

        for (const segment of segments) {
            for (const { re, type } of patterns) {
                if (re.test(segment)) {
                    return type;
                }
            }
        }

        return null;
    }

    renderModels(repoId, models) {
        this.modelsTbody.innerHTML = "";

        if (models.length === 0) {
            this.modelsTbody.innerHTML = '<tr><td colspan="9">No supported model files found</td></tr>';
            this.modelsSection.style.display = "block";
            return;
        }

        models.forEach((model, index) => {
            const row = document.createElement("tr");
            const fileEntries = model.files.map(f => typeof f === 'string' ? { name: f } : f);
            const isGguf = model.file_type === "gguf" || fileEntries.some(f => f.name.toLowerCase().endsWith(".gguf"));
            const isSplit = Boolean(model.is_split);
            const isSelectable = !isSplit && fileEntries.length > 1;

            const typeBadge = model.is_split ?
                '<span class="badge badge-split">Split Files</span>' :
                (isSelectable ? '<span class="badge badge-single">Variants</span>' : '<span class="badge badge-single">Single File</span>');

            const defaultSizeBytes = (isGguf || isSelectable) ? (fileEntries[0]?.size || 0) : model.total_size;
            const sizeFormatted = this.formatSize(defaultSizeBytes);
            let precisionLabel = model.precision;
            if (!precisionLabel && !isGguf && isSelectable) {
                const precisionValues = [...new Set(fileEntries.map(f => f.precision).filter(Boolean))];
                if (precisionValues.length === 1) {
                    precisionLabel = precisionValues[0];
                } else if (precisionValues.length > 1) {
                    precisionLabel = "mixed";
                }
            }
            const precisionBadge = precisionLabel ?
                `<span class="badge badge-precision">${precisionLabel.toUpperCase()}</span>` :
                (isGguf ? '<span class="badge badge-precision">GGUF</span>' : '<span class="badge badge-unknown">Unknown</span>');

            let quantSelectHtml = '<span class="badge badge-unknown">N/A</span>';
            if (isGguf || isSelectable) {
                const usedLabels = new Set();
                const options = fileEntries.map((file, fileIndex) => {
                    const quantValue = file.quant ? file.quant.toUpperCase() : "";
                    const variantValue = file.variant || file.precision || "";
                    const displayVariant = variantValue ? variantValue.replace(/_/g, "-").toUpperCase() : "";
                    let label = quantValue || displayVariant || file.name;
                    if (usedLabels.has(label)) {
                        label = file.name;
                    }
                    usedLabels.add(label);
                    const sizeValue = Number.isFinite(file.size) ? String(file.size) : "0";
                    return `<option value="${file.name}" data-quant="${quantValue}" data-variant="${variantValue}" data-size="${sizeValue}" ${fileIndex === 0 ? "selected" : ""}>${label}</option>`;
                }).join("");
                quantSelectHtml = `<select class="quant-select" id="quant-${index}">${options}</select>`;
            }

            const modelLabelBase = model.suggested_name || model.base_name || model.path;
            const modelLabel = (model.path && model.path !== "root" && modelLabelBase) ?
                `${model.path}/${modelLabelBase}` :
                (modelLabelBase || model.path);

            row.innerHTML = `
                <td>${modelLabel}</td>
                <td>${typeBadge}</td>
                <td>${model.file_count}</td>
                <td id="size-${index}" class="size-cell">${sizeFormatted}</td>
                <td>${precisionBadge}</td>
                <td>${quantSelectHtml}</td>
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
                        <option value="custom">Custom...</option>
                    </select>
                    <input type="text" class="custom-path-input" id="custom-path-${index}" placeholder="e.g. custom_nodes/MyNode/models" style="display: none; margin-top: 4px;" />
                </td>
                <td><button class="download-btn" data-index="${index}">Download</button></td>
            `;

            const downloadBtn = row.querySelector(".download-btn");
            downloadBtn.onclick = () => this.downloadModel(repoId, model, index);

            const outputNameInput = row.querySelector(".output-name-input");
            outputNameInput.dataset.autoname = "true";
            outputNameInput.addEventListener("input", () => {
                outputNameInput.dataset.autoname = "false";
            });

            const quantSelect = row.querySelector(".quant-select");
            if ((isGguf || isSelectable) && quantSelect) {
                const baseName = model.base_name || model.suggested_name || "model";
                const sizeCell = row.querySelector(`#size-${index}`);
                const updateSelection = () => {
                    const selectedOption = quantSelect.options[quantSelect.selectedIndex];
                    const sizeBytes = parseInt(selectedOption?.dataset.size || "0", 10);
                    if (sizeCell) {
                        sizeCell.textContent = this.formatSize(sizeBytes);
                    }
                    if (outputNameInput.dataset.autoname !== "true") {
                        return;
                    }
                    let suffix = "";
                    if (isGguf) {
                        suffix = selectedOption ? selectedOption.dataset.quant : "";
                    } else if (isSelectable) {
                        suffix = selectedOption ? selectedOption.dataset.variant : "";
                    }
                    outputNameInput.value = suffix ? `${baseName}-${suffix}` : baseName;
                };
                quantSelect.addEventListener("change", updateSelection);
                updateSelection();
            }

            const modelTypeSelect = row.querySelector(`#type-${index}`);
            const customPathInput = row.querySelector(`#custom-path-${index}`);
            const inferredType = this.inferModelTypeFromPath(model.path);
            if (modelTypeSelect && inferredType) {
                modelTypeSelect.value = inferredType;
            }

            if (modelTypeSelect && customPathInput) {
                modelTypeSelect.addEventListener("change", () => {
                    customPathInput.style.display = modelTypeSelect.value === "custom" ? "block" : "none";
                });
            }

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

        // Validate custom path if selected
        let customPath = null;
        if (modelType === "custom") {
            const customPathInput = document.getElementById(`custom-path-${index}`);
            customPath = customPathInput ? customPathInput.value.trim() : "";
            if (!customPath) {
                alert("Please enter a custom path (relative to models directory)");
                return;
            }
        }

        const fileEntries = model.files.map(f => typeof f === 'string' ? { name: f } : f);
        const isGguf = model.file_type === "gguf" || fileEntries.some(f => f.name.toLowerCase().endsWith(".gguf"));
        const isSplit = Boolean(model.is_split);
        const isSelectable = !isSplit && fileEntries.length > 1;

        // Extract filenames from file metadata objects
        // model.files is now an array of {name, size, precision} objects
        let filenames = fileEntries.map(f => f.name);
        if (isGguf || isSelectable) {
            const quantSelect = document.getElementById(`quant-${index}`);
            if (quantSelect && quantSelect.value) {
                filenames = [quantSelect.value];
            }
        }

        const requestBody = {
            repo_id: repoId,
            model_path: model.path,
            files: filenames,
            output_name: outputName,
            model_type: modelType
        };
        if (customPath) {
            requestBody.custom_path = customPath;
        }

        try {
            const response = await api.fetchApi("/hf_downloader/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
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
            <div class="progress-actions" id="actions-${taskId}" style="display: flex; gap: 10px; margin-top: 10px;">
                <button class="action-btn abort-btn" id="abort-${taskId}">Abort</button>
            </div>
        `;

        this.progressSection.appendChild(progressDiv);
        this.currentTasks.set(taskId, { div: progressDiv, modelName });

        // Add abort button handler
        const abortBtn = document.getElementById(`abort-${taskId}`);
        if (abortBtn) {
            abortBtn.onclick = async () => {
                if (confirm(`Abort download of ${modelName}?`)) {
                    abortBtn.disabled = true;
                    abortBtn.textContent = "Aborting...";
                    await this.abortDownload(taskId);
                }
            };
        }

        const pollInterval = setInterval(async () => {
            try {
                const response = await api.fetchApi(`/hf_downloader/progress/${taskId}`);
                const progress = await response.json();

                this.updateProgress(taskId, progress);

                if (progress.status === "completed" || progress.status === "error" || progress.status === "cancelled") {
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

        // Debug logging
        console.log(`[HF Downloader] Progress update for ${taskId}:`, progress);

        if (stageEl) {
            stageEl.textContent = progress.stage.toUpperCase();
            // Color coding for status
            if (progress.status === "error") {
                stageEl.style.color = "#ff6b6b";
            } else if (progress.status === "completed") {
                stageEl.style.color = "#4ade80";  // Green for success
            } else {
                stageEl.style.color = "";  // Reset to default
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

    switchTab(tabName) {
        // Update tab buttons
        this.modal.querySelectorAll(".hf-tab").forEach(tab => {
            tab.classList.toggle("active", tab.dataset.tab === tabName);
        });

        // Update tab contents
        this.modal.querySelectorAll(".hf-tab-content").forEach(content => {
            content.classList.toggle("active", content.id === `hf-tab-${tabName}`);
        });

        if (tabName === "manage") {
            this.loadFiles();
        }

        if (tabName === "status") {
            this.startStatusPolling();
        } else {
            this.stopStatusPolling();
        }
    }

    startStatusPolling() {
        if (this.statusInterval) return;
        this.refreshStatus();
        this.statusInterval = setInterval(() => this.refreshStatus(), 2000);
    }

    stopStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    async refreshStatus() {
        try {
            const response = await api.fetchApi("/hf_downloader/status");
            const items = await response.json();
            if (!Array.isArray(items)) return;
            this.renderStatus(items);
            this.updateStatusBadge(items);
        } catch (e) {
            console.error("[HF Downloader] Status fetch error:", e);
        }
    }

    updateStatusBadge(items) {
        const active = items.filter(i => i.status === "running" || i.status === "queued" || i.status === "starting").length;
        if (this.statusBadge) {
            this.statusBadge.textContent = active;
            this.statusBadge.style.display = active > 0 ? "inline-block" : "none";
        }
    }

    renderStatus(items) {
        const cleared = this._clearedItems || new Set();
        const active = items.filter(i => ["running", "queued", "starting"].includes(i.status));
        const done   = items.filter(i => i.status === "completed" && !cleared.has(i.task_id));
        const failed = items.filter(i => ["error", "cancelled"].includes(i.status) && !cleared.has(i.task_id));

        this.renderStatusGroup(this.statusActiveList, this.statusActiveSection, active, "Nothing active or queued.");
        this.renderStatusGroup(this.statusDoneList,   this.statusDoneSection,   done,   "No completed downloads yet.");
        this.renderStatusGroup(this.statusFailedList, this.statusFailedSection, failed, "No failed downloads.");
    }

    renderStatusGroup(container, section, items, emptyMsg) {
        if (!container || !section) return;

        if (items.length === 0) {
            section.style.display = "none";
            return;
        }
        section.style.display = "block";

        container.innerHTML = "";
        items.forEach(item => {
            const tid = item.task_id || "";
            const name = item.name || tid;
            const status = item.status || "unknown";
            const msg = item.message || "";
            const pct = item.total > 0 ? Math.round(item.current / item.total * 100) : 0;
            const isIndeterminate = status === "running" && item.total === 0;

            let barClass = "";
            if (status === "completed") barClass = "hf-sbar-done";
            else if (status === "error" || status === "cancelled") barClass = "hf-sbar-failed";

            const barStyle = isIndeterminate
                ? 'style="width:100%" class="hf-status-bar hf-sbar-indeterminate"'
                : `style="width:${status === "completed" ? 100 : pct}%" class="hf-status-bar ${barClass}"`;

            const el = document.createElement("div");
            el.className = "hf-status-item";
            el.dataset.tid = tid;
            el.innerHTML = `
                <div class="hf-status-item-header">
                    <span class="hf-status-item-name" title="${name}">${name}</span>
                    <span class="hf-status-pill hf-pill-${status}">${status}</span>
                </div>
                <div class="hf-status-bar-track">
                    <div ${barStyle}></div>
                </div>
                ${msg ? `<div class="hf-status-item-msg">${msg}</div>` : ""}
                ${(status === "running" || status === "queued" || status === "starting") && tid
                    ? `<button class="hf-status-abort-btn" data-tid="${tid}">Abort</button>`
                    : ""}
            `;

            const abortBtn = el.querySelector(".hf-status-abort-btn");
            if (abortBtn) {
                abortBtn.onclick = async () => {
                    abortBtn.disabled = true;
                    abortBtn.textContent = "Aborting…";
                    await this.abortDownload(tid);
                    this.refreshStatus();
                };
            }

            container.appendChild(el);
        });
    }

    clearDoneItems() {
        // Track which task_ids have been dismissed client-side
        if (!this._clearedItems) this._clearedItems = new Set();
        // Add all currently completed/failed items to the cleared set
        this.statusDoneList.querySelectorAll(".hf-status-item").forEach(el => {
            const tid = el.dataset.tid;
            if (tid) this._clearedItems.add(tid);
        });
        this.statusFailedList.querySelectorAll(".hf-status-item").forEach(el => {
            const tid = el.dataset.tid;
            if (tid) this._clearedItems.add(tid);
        });
        this.refreshStatus();
    }

    async loadFiles() {
        const modelType = this.fileTypeFilter.value;
        this.filesList.innerHTML = '<div class="loading">Loading files...</div>';

        try {
            const response = await api.fetchApi(`/hf_downloader/files/${modelType}`);
            const data = await response.json();

            if (!data.files || data.files.length === 0) {
                this.filesList.innerHTML = '<div class="no-files">No files found in this category</div>';
                return;
            }

            // Build file list table
            let html = `
                <table class="hf-files-table">
                    <thead>
                        <tr>
                            <th>File Name</th>
                            <th>Size</th>
                            <th>Modified</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            for (const file of data.files) {
                const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                const date = new Date(file.modified * 1000).toLocaleString();

                html += `
                    <tr>
                        <td title="${file.path}">${file.name}</td>
                        <td>${sizeMB} MB</td>
                        <td>${date}</td>
                        <td>
                            <button class="delete-file-btn" data-filepath="${file.path}" data-filename="${file.name}">Delete</button>
                        </td>
                    </tr>
                `;
            }

            html += '</tbody></table>';
            this.filesList.innerHTML = html;

            // Attach delete handlers
            this.filesList.querySelectorAll('.delete-file-btn').forEach(btn => {
                btn.onclick = () => this.deleteFile(btn.dataset.filepath, btn.dataset.filename);
            });

        } catch (error) {
            console.error("[HF Downloader] Error loading files:", error);
            this.filesList.innerHTML = '<div class="error">Error loading files. Check console for details.</div>';
        }
    }

    async deleteFile(filepath, filename) {
        if (!confirm(`Delete ${filename}?\n\nThis cannot be undone.`)) {
            return;
        }

        try {
            const response = await api.fetchApi('/hf_downloader/files', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filepath })
            });

            const data = await response.json();

            if (data.success) {
                // Reload file list
                this.loadFiles();
                console.log(`[HF Downloader] Deleted: ${filename}`);
            } else {
                alert(`Error deleting file: ${data.error}`);
            }
        } catch (error) {
            console.error("[HF Downloader] Error deleting file:", error);
            alert('Error deleting file. Check console for details.');
        }
    }

    async abortDownload(taskId) {
        try {
            const response = await api.fetchApi(`/hf_downloader/abort/${taskId}`, {
                method: 'POST'
            });

            const data = await response.json();
            console.log(`[HF Downloader] Abort response:`, data);
        } catch (error) {
            console.error("[HF Downloader] Error aborting download:", error);
        }
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
            width: calc(100% - 48px);
            max-width: none;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
            box-sizing: border-box;
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

        .badge-precision {
            background: #4a9eff;
            color: #fff;
        }

        .badge-unknown {
            background: #666;
            color: #ccc;
        }

        .output-name-input,
        .model-type-select,
        .quant-select,
        .custom-path-input {
            padding: 6px;
            background: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            border-radius: 3px;
            width: 100%;
        }

        .custom-path-input {
            font-size: 12px;
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

        .hf-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            border-bottom: 2px solid #444;
        }

        .hf-tab {
            padding: 10px 20px;
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }

        .hf-tab:hover {
            color: #fff;
        }

        .hf-tab.active {
            color: #4a9eff;
            border-bottom-color: #4a9eff;
        }

        .hf-tab-content {
            display: none;
        }

        .hf-tab-content.active {
            display: block;
        }

        .hf-manage-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 4px;
        }

        .hf-manage-controls label {
            font-weight: bold;
        }

        #hf-file-type-filter {
            padding: 8px;
            background: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            border-radius: 4px;
            flex: 1;
        }

        #hf-refresh-files {
            padding: 8px 16px;
            background: #4a9eff;
            border: none;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
        }

        #hf-refresh-files:hover {
            background: #3a8eef;
        }

        .hf-files-table {
            width: 100%;
            border-collapse: collapse;
            background: #2a2a2a;
        }

        .hf-files-table th,
        .hf-files-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }

        .hf-files-table th {
            background: #1a1a1a;
            font-weight: bold;
            position: sticky;
            top: 0;
        }

        .hf-files-table tr:hover {
            background: #333;
        }

        .delete-file-btn {
            padding: 6px 12px;
            background: #ff4444;
            border: none;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
        }

        .delete-file-btn:hover {
            background: #cc0000;
        }

        .loading, .no-files, .error {
            padding: 20px;
            text-align: center;
            color: #888;
        }

        .error {
            color: #ff6b6b;
        }

        .abort-btn {
            background: #ff6b6b !important;
        }

        .abort-btn:hover {
            background: #ff4444 !important;
        }

        .abort-btn:disabled {
            background: #666 !important;
            cursor: not-allowed;
        }

        /* ── Status tab ─────────────────────────────────────────────────── */

        .hf-status-badge {
            display: inline-block;
            background: #e53e3e;
            color: #fff;
            font-size: 11px;
            font-weight: bold;
            border-radius: 10px;
            padding: 1px 6px;
            margin-left: 4px;
            vertical-align: middle;
            line-height: 16px;
        }

        .hf-status-toolbar {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .hf-status-toolbar button {
            padding: 6px 14px;
            background: #333;
            border: 1px solid #555;
            color: #ccc;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }

        .hf-status-toolbar button:hover {
            background: #444;
            color: #fff;
        }

        .hf-status-section-label {
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #777;
            margin-bottom: 8px;
        }

        #hf-status-active-section,
        #hf-status-done-section,
        #hf-status-failed-section {
            margin-bottom: 20px;
        }

        .hf-status-item {
            background: #2a2a2a;
            border-radius: 5px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }

        .hf-status-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            gap: 8px;
        }

        .hf-status-item-name {
            font-size: 13px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
            min-width: 0;
        }

        .hf-status-pill {
            flex-shrink: 0;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            border-radius: 3px;
            padding: 2px 7px;
        }

        .hf-pill-running   { background: #1c4fa0; color: #90c4ff; }
        .hf-pill-queued    { background: #5a3d00; color: #ffc947; }
        .hf-pill-starting  { background: #1c4fa0; color: #90c4ff; }
        .hf-pill-completed { background: #1a4a2a; color: #4ade80; }
        .hf-pill-error     { background: #4a1a1a; color: #ff6b6b; }
        .hf-pill-cancelled { background: #3a3a3a; color: #aaa; }
        .hf-pill-unknown   { background: #333;    color: #888; }

        .hf-status-bar-track {
            width: 100%;
            height: 6px;
            background: #1a1a1a;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 6px;
        }

        .hf-status-bar {
            height: 100%;
            background: linear-gradient(90deg, #4a9eff 0%, #3a8eef 100%);
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .hf-sbar-done   { background: linear-gradient(90deg, #51cf66 0%, #40bf56 100%); }
        .hf-sbar-failed { background: linear-gradient(90deg, #ff6b6b 0%, #ef5b5b 100%); }

        @keyframes hf-indeterminate {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(400%); }
        }

        .hf-sbar-indeterminate {
            width: 25% !important;
            animation: hf-indeterminate 1.4s ease infinite;
        }

        .hf-status-item-msg {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
            word-break: break-all;
        }

        .hf-status-abort-btn {
            margin-top: 4px;
            padding: 3px 10px;
            background: #6b1a1a;
            border: none;
            color: #ff9999;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }

        .hf-status-abort-btn:hover  { background: #8b2020; }
        .hf-status-abort-btn:disabled { background: #444; color: #777; cursor: not-allowed; }
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
