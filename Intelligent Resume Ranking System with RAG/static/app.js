document.addEventListener('DOMContentLoaded', () => {

    // Upload Logic
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('resume-files');
    const fileListMsg = document.querySelector('.file-msg');
    const fileListDiv = document.getElementById('file-list');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const evalCollectionSelect = document.getElementById('eval-collection');

    // Load collections on startup
    async function loadCollections() {
        try {
            const res = await fetch('/api/v1/collections');
            const data = await res.json();
            if (data.collections) {
                evalCollectionSelect.innerHTML = data.collections.map(c =>
                    `<option value="${c}">${c}</option>`
                ).join('');
            }
        } catch (e) {
            console.error("Failed to load collections", e);
            evalCollectionSelect.innerHTML = `<option value="resume_chunks">resume_chunks</option>`;
        }
    }

    loadCollections();

    let selectedFiles = [];

    // Drag and Drop styling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        updateFileList();
    }

    function updateFileList() {
        if (selectedFiles.length > 0) {
            const totalMB = selectedFiles.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
            fileListMsg.textContent = `${selectedFiles.length} file(s) selected`;
            fileListDiv.innerHTML = `<div class="file-summary">${selectedFiles.length} files &middot; ${totalMB.toFixed(2)} MB total</div>`;
        } else {
            fileListMsg.textContent = "Drag & drop PDFs/DOCXs here or click to browse";
            fileListDiv.innerHTML = "";
        }
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0) return;

        const formData = new FormData();
        const collectionName = document.getElementById('collection-name').value.trim();
        if (collectionName) {
            formData.append('collection_name', collectionName);
        }

        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Ingesting...';
        uploadStatus.className = 'status-msg hidden';

        // Show progress bar
        fileListDiv.innerHTML = `
            <div class="upload-progress">
                <div class="progress-bar"><div class="progress-fill"></div></div>
                <span class="progress-text">Processing ${selectedFiles.length} files...</span>
            </div>
        `;
        const fill = fileListDiv.querySelector('.progress-fill');
        fill.style.width = '80%';

        try {
            const response = await fetch('/api/v1/upload-resumes', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                uploadStatus.textContent = `Success: Ingested ${data.processed_count} resumes!`;
                uploadStatus.className = 'status-msg success';
                selectedFiles = [];
                updateFileList();
                loadCollections();
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = 'status-msg error';
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Ingest Resumes';
        }
    });

    // Evaluation Logic
    const evaluateForm = document.getElementById('evaluate-form');
    const evalBtn = document.getElementById('evaluate-btn');
    const evalLoader = document.getElementById('eval-loader');
    const resultsContainer = document.getElementById('results-container');
    const template = document.getElementById('result-card-template');

    evaluateForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const jdText = document.getElementById('jd-text').value.trim();
        const topK = parseInt(document.getElementById('top-k').value, 10);
        const selectedCollection = evalCollectionSelect.value;

        if (!jdText) return;

        evalBtn.disabled = true;
        evalLoader.classList.remove('hidden');
        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('empty-state');

        try {
            const response = await fetch('/api/v1/evaluate-job', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_description: jdText,
                    collection_name: selectedCollection,
                    top_k: topK
                })
            });

            const data = await response.json();

            if (response.ok) {
                renderResults(data.rankings);
            } else {
                throw new Error(data.detail || 'Evaluation failed');
            }
        } catch (error) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <p style="color:var(--danger)">Error: ${error.message}</p>
                </div>
            `;
            resultsContainer.classList.add('empty-state');
        } finally {
            evalBtn.disabled = false;
            evalLoader.classList.add('hidden');
        }
    });

    function renderResults(rankings) {
        if (!rankings || rankings.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <p>No matches found.</p>
                </div>
            `;
            resultsContainer.classList.add('empty-state');
            return;
        }

        rankings.forEach((rank, index) => {
            const clone = template.content.cloneNode(true);

            // Set candidate name
            clone.querySelector('.candidate-id').textContent = rank.candidate_id;

            // Format score
            let scorePercent = Math.round(rank.overall_score * 100);
            if (isNaN(scorePercent)) scorePercent = 0;
            const scoreCircle = clone.querySelector('.score-circle');
            clone.querySelector('.score-val').textContent = `${scorePercent}`;

            // Color grading based on score
            if (scorePercent >= 80) {
                scoreCircle.style.borderColor = '#fff';
                scoreCircle.style.color = '#fff';
            } else if (scorePercent >= 50) {
                scoreCircle.style.borderColor = '#888';
                scoreCircle.style.color = '#ccc';
            } else {
                scoreCircle.style.borderColor = '#444';
                scoreCircle.style.color = '#888';
            }

            // Score breakdown text
            const breakdown = clone.querySelector('.score-breakdown');
            if (rank.score_details) {
                breakdown.textContent = `Sem: ${Math.round(rank.score_details.semantic * 100)}% · Skills: ${Math.round(rank.score_details.skills * 100)}% · Exp: ${Math.round(rank.score_details.experience * 100)}%`;
            } else {
                breakdown.textContent = `Score: ${scorePercent}%`;
            }

            // Stagger animation
            const targetDiv = clone.querySelector('.result-card');
            targetDiv.style.animationDelay = `${index * 0.1}s`;

            resultsContainer.appendChild(clone);
        });
    }

});
