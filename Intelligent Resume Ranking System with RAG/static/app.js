document.addEventListener('DOMContentLoaded', () => {

    // DOM refs
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('resume-files');
    const fileListMsg = document.querySelector('.file-msg');
    const fileListDiv = document.getElementById('file-list');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const collectionsListDiv = document.getElementById('collections-list');
    const evalCollectionInput = document.getElementById('eval-collection');

    let selectedFiles = [];
    let activeCollection = 'resume_chunks';

    // ─── Collections ───

    async function loadCollections() {
        try {
            const res = await fetch('/api/v1/collections');
            const data = await res.json();
            if (data.collections && data.collections.length > 0) {
                renderCollections(data.collections);
            } else {
                collectionsListDiv.innerHTML = '<p class="text-muted">No collections yet</p>';
            }
        } catch (e) {
            console.error("Failed to load collections", e);
            collectionsListDiv.innerHTML = '<p class="text-muted">Could not load</p>';
        }
    }

    function renderCollections(collections) {
        collectionsListDiv.innerHTML = '';
        collections.forEach(name => {
            const item = document.createElement('div');
            item.className = 'collection-item' + (name === activeCollection ? ' active' : '');
            item.innerHTML = `<span>${name}</span><span class="col-check">✓</span>`;
            item.addEventListener('click', () => selectCollection(name));
            collectionsListDiv.appendChild(item);
        });
    }

    function selectCollection(name) {
        activeCollection = name;
        evalCollectionInput.value = name;
        // Update active state visually
        document.querySelectorAll('.collection-item').forEach(el => {
            el.classList.toggle('active', el.querySelector('span').textContent === name);
        });
    }

    loadCollections();

    // ─── File Upload (Drag & Drop) ───

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
        dropArea.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(ev => {
        dropArea.addEventListener(ev, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(ev => {
        dropArea.addEventListener(ev, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function () { handleFiles(this.files); });

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        updateFileList();
    }

    function updateFileList() {
        if (selectedFiles.length > 0) {
            const totalMB = selectedFiles.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
            fileListMsg.textContent = `${selectedFiles.length} file(s) selected`;
            fileListDiv.innerHTML = `<div class="file-summary">${selectedFiles.length} files · ${totalMB.toFixed(2)} MB total</div>`;
        } else {
            fileListMsg.textContent = "Drag & drop PDFs/DOCXs here or click to browse";
            fileListDiv.innerHTML = "";
        }
    }

    // ─── Upload Submit ───

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0) return;

        const formData = new FormData();
        const collectionName = document.getElementById('collection-name').value.trim();
        if (collectionName) {
            formData.append('collection_name', collectionName);
        }

        selectedFiles.forEach(file => formData.append('files', file));

        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Ingesting...';
        uploadStatus.className = 'status-msg hidden';

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
                uploadStatus.textContent = `Ingested ${data.processed_count} resumes`;
                uploadStatus.className = 'status-msg success';
                selectedFiles = [];
                updateFileList();
                // Refresh collections & auto-select the one just uploaded to
                activeCollection = collectionName || 'resume_chunks';
                evalCollectionInput.value = activeCollection;
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

    // ─── Evaluation ───

    const evaluateForm = document.getElementById('evaluate-form');
    const evalBtn = document.getElementById('evaluate-btn');
    const evalLoader = document.getElementById('eval-loader');
    const resultsContainer = document.getElementById('results-container');
    const template = document.getElementById('result-card-template');

    evaluateForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const jdText = document.getElementById('jd-text').value.trim();
        const topK = parseInt(document.getElementById('top-k').value, 10);

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
                    collection_name: activeCollection,
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
                    <p style="color:#999">Error: ${error.message}</p>
                </div>
            `;
            resultsContainer.classList.add('empty-state');
        } finally {
            evalBtn.disabled = false;
            evalLoader.classList.add('hidden');
        }
    });

    // ─── Render Results ───

    function stripId(candidateId) {
        // candidate_id format: "a1b2c3d4_filename.pdf"
        // Strip everything up to and including the first underscore
        const idx = candidateId.indexOf('_');
        return idx !== -1 ? candidateId.substring(idx + 1) : candidateId;
    }

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

            // Rank number
            clone.querySelector('.rc-rank').textContent = `#${index + 1}`;

            // Set candidate name — strip UUID prefix, show only filename
            clone.querySelector('.candidate-id').textContent = stripId(rank.candidate_id);

            // Score
            let scorePercent = Math.round(rank.overall_score * 100);
            if (isNaN(scorePercent)) scorePercent = 0;
            const scoreCircle = clone.querySelector('.score-circle');
            clone.querySelector('.score-val').textContent = `${scorePercent}`;

            // Monochrome grading
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

            // Score breakdown
            const breakdown = clone.querySelector('.score-breakdown');
            if (rank.score_details) {
                breakdown.textContent = `Sem: ${Math.round(rank.score_details.semantic * 100)}%  ·  Skills: ${Math.round(rank.score_details.skills * 100)}%  ·  Exp: ${Math.round(rank.score_details.experience * 100)}%`;
            } else {
                breakdown.textContent = `Score: ${scorePercent}%`;
            }

            // Stagger animation
            clone.querySelector('.result-card').style.animationDelay = `${index * 0.08}s`;

            resultsContainer.appendChild(clone);
        });
    }

});
