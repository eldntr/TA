document.addEventListener('DOMContentLoaded', () => {
    const METADATA_URL = '/create-dataset/id/final_dataset/metadata.csv';
    const AUDIO_RESTORED_URL = '/create-dataset/id/final_dataset/wavs/';
    const AUDIO_ORIGINAL_URL = '/create-dataset/id/final_dataset/wavs/';
    
    let dataset = [];
    
    const ui = {
        status: document.getElementById('status-badge'),
        filename: document.getElementById('filename-display'),
        transcript: document.getElementById('transcript-display'),
        playBtnOriginal: document.getElementById('play-btn-original'),
        playBtnRestored: document.getElementById('play-btn-restored'),
        shuffleBtn: document.getElementById('shuffle-btn'),
        totalCount: document.getElementById('total-count')
    };

    const wsOptions = {
        waveColor: 'rgba(255, 255, 255, 0.4)',
        progressColor: '#3b82f6',
        barWidth: 2,
        barGap: 2,
        barRadius: 2,
        height: 40,
        normalize: true
    };
    
    let wavesurferOriginal = WaveSurfer.create({
        container: '#waveform-original',
        ...wsOptions
    });
    
    let wavesurferRestored = WaveSurfer.create({
        container: '#waveform-restored',
        ...wsOptions,
        progressColor: '#10b981' // Hijau untuk restored
    });

    // Handle Play/Pause
    const playIcon = '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    const pauseIcon = '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';

    ui.playBtnOriginal.addEventListener('click', () => wavesurferOriginal.playPause());
    ui.playBtnRestored.addEventListener('click', () => wavesurferRestored.playPause());
    
    wavesurferOriginal.on('play', () => ui.playBtnOriginal.innerHTML = pauseIcon);
    wavesurferOriginal.on('pause', () => ui.playBtnOriginal.innerHTML = playIcon);
    
    wavesurferRestored.on('play', () => ui.playBtnRestored.innerHTML = pauseIcon);
    wavesurferRestored.on('pause', () => ui.playBtnRestored.innerHTML = playIcon);

    // Initialize: Fetch metadata
    async function init() {
        try {
            ui.status.textContent = "Mengunduh Metadata...";
            const response = await fetch(METADATA_URL);
            
            if (!response.ok) {
                throw new Error("Gagal mengambil metadata.csv. Pastikan server berjalan.");
            }
            
            const text = await response.text();
            parseMetadata(text);
            
            ui.status.textContent = "Siap Digunakan";
            ui.status.className = "status-badge success";
            ui.shuffleBtn.disabled = false;
            
            // Auto play first random
            playRandom();
            
        } catch (error) {
            console.error(error);
            ui.status.textContent = "Error: " + error.message;
            ui.status.className = "status-badge error";
        }
    }

    function parseMetadata(csvText) {
        const lines = csvText.split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            
            const firstPipeIdx = trimmed.indexOf('|');
            if (firstPipeIdx > 0) {
                const filename = trimmed.substring(0, firstPipeIdx).trim();
                const transcript = trimmed.substring(firstPipeIdx + 1).trim();
                dataset.push({ filename, transcript });
            }
        }
        ui.totalCount.textContent = dataset.length.toLocaleString();
        console.log(`Parsed ${dataset.length} items from metadata.`);
    }

    function playRandom() {
        if (dataset.length === 0) return;
        
        ui.transcript.style.opacity = 0;
        ui.status.textContent = "Mencari file audio bersih...";
        ui.status.className = "status-badge";
        ui.shuffleBtn.disabled = true;
        
        attemptPlayRandom();
    }

    async function attemptPlayRandom() {
        const randomIndex = Math.floor(Math.random() * dataset.length);
        const item = dataset[randomIndex];
        
        const audioUrlRestored = AUDIO_RESTORED_URL + item.filename;
        const audioUrlOriginal = AUDIO_ORIGINAL_URL + item.filename;
        
        try {
            // Check if restored file exists (avoid wavesurfer error loops)
            const res = await fetch(audioUrlRestored, { method: 'HEAD' });
            if (!res.ok) throw new Error("File not found");
            
            ui.filename.textContent = item.filename;
            ui.transcript.textContent = item.transcript;
            
            setTimeout(() => {
                ui.transcript.style.opacity = 1;
            }, 100);
            
            ui.status.textContent = "Audio Bersih Ditemukan";
            ui.status.className = "status-badge success";
            ui.shuffleBtn.disabled = false;
            
            // Load audio into wave surfers
            wavesurferRestored.load(audioUrlRestored);
            wavesurferOriginal.load(audioUrlOriginal);
            
            // Autoplay the restored one when ready
            wavesurferRestored.once('ready', () => {
                const playPromise = wavesurferRestored.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        console.log("Autoplay dicegah browser.");
                    });
                }
            });
            
        } catch(e) {
            console.log(`Audio dibuang/tidak ada: ${item.filename}, mencari yang lain...`);
            attemptPlayRandom();
        }
    }

    ui.shuffleBtn.addEventListener('click', playRandom);
    
    // Mulai inisialisasi
    init();
});
