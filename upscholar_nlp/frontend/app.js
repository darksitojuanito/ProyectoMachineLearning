document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const input = document.getElementById('query-input');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const queryDisplay = document.getElementById('query-display');
    const resultsList = document.getElementById('results-list');
    
    // Menu elements
    const menuDataset = document.getElementById('menu-dataset');
    const menuEmbeddings = document.getElementById('menu-embeddings');
    
    let currentMode = 'dataset'; // 'dataset' or 'embeddings'

    // Switch mode to Dataset
    menuDataset.addEventListener('click', (e) => {
        e.preventDefault();
        currentMode = 'dataset';
        menuDataset.classList.add('active');
        menuEmbeddings.classList.remove('active');
        input.placeholder = "Ej: deep learning medical image segmentation";
        resultsContainer.classList.add('hidden');
        resultsList.innerHTML = '';
        input.value = '';
    });

    // Switch mode to Embeddings
    menuEmbeddings.addEventListener('click', (e) => {
        e.preventDefault();
        currentMode = 'embeddings';
        menuEmbeddings.classList.add('active');
        menuDataset.classList.remove('active');
        input.placeholder = "Buscar por significado usando abstracts...";
        resultsContainer.classList.add('hidden');
        resultsList.innerHTML = '';
        input.value = '';
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = input.value.trim();
        if (!query) {
            alert("La consulta no puede estar vacía. Intenta nuevamente.");
            return;
        }

        // UI State: Loading
        loader.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        resultsList.innerHTML = '';

        try {
            const endpoint = currentMode === 'dataset' ? '/api/search' : '/api/search-embeddings';
            
            const fetchPromise = fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            const delayPromise = new Promise(resolve => setTimeout(resolve, 1000));

            const [response] = await Promise.all([fetchPromise, delayPromise]);

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Error en la búsqueda.');
            }

            // Render Results
            if (currentMode === 'dataset') {
                renderDatasetResults(data);
            } else {
                renderEmbeddingResults(data);
            }

        } catch (error) {
            alert(error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    function renderQueryInfo(data) {
        const queryInfoContainer = document.getElementById('query-info-container');
        if (data.original_query.toLowerCase() !== data.processed_query.toLowerCase()) {
            let detailsHtml = '';
            if (data.query_processing.was_translated) {
                detailsHtml += `<div>Consulta traducida al inglés para mejorar la búsqueda en artículos científicos.</div>`;
            }
            if (data.query_processing.was_corrected && data.query_processing.corrections.length > 0) {
                const correctionsStr = data.query_processing.corrections.map(c => `${c.original} → ${c.corrected}`).join(', ');
                detailsHtml += `<div>Correcciones aplicadas: ${correctionsStr}</div>`;
            }
            if (data.query_processing.used_wildcards && data.query_processing.wildcard_expansions.length > 0) {
                const wildcardsStr = data.query_processing.wildcard_expansions.map(w => `${w.pattern} → ${w.matches.join(', ')}`).join(' | ');
                detailsHtml += `<div>Comodines expandidos: ${wildcardsStr}</div>`;
            }
            
            queryInfoContainer.innerHTML = `
                <div class="query-info">
                    <div>Se muestran resultados para: <strong>${data.processed_query}</strong></div>
                    <div class="query-details">Consulta original: ${data.original_query}</div>
                    <div class="query-details">${detailsHtml}</div>
                </div>
            `;
        } else {
            queryInfoContainer.innerHTML = '';
        }
    }

    function renderDatasetResults(data) {
        queryDisplay.textContent = data.processed_query;
        renderQueryInfo(data);
        
        if (data.has_results === false) {
            resultsList.innerHTML = `
                <div class="no-results">
                    <h3>${data.message || "No se encontró información relacionada con tu consulta"}</h3>
                    <p>El dataset contiene artículos científicos de ICMLA 2020 sobre temas de machine learning y aplicaciones relacionadas. Intenta usar términos más cercanos al contenido del dataset.</p>
                    <div class="suggestion-tags">
                        <span class="suggestion-tag">medical image segmentation</span>
                        <span class="suggestion-tag">graph neural networks</span>
                        <span class="suggestion-tag">biometric face recognition</span>
                        <span class="suggestion-tag">reinforcement learning</span>
                        <span class="suggestion-tag">energy forecasting</span>
                        <span class="suggestion-tag">computer vision</span>
                        <span class="suggestion-tag">natural language processing</span>
                    </div>
                </div>
            `;
            resultsContainer.classList.remove('hidden');
            return;
        }
        
        if (!data.top10 || data.top10.length === 0) {
            resultsList.innerHTML = '<p>No se encontraron resultados.</p>';
            resultsContainer.classList.remove('hidden');
            return;
        }

        const html = data.top10.map(paper => {
            const abstractSnippet = paper.abstract.length > 250 
                ? paper.abstract.substring(0, 250) + '...' 
                : paper.abstract;

            const recsHtml = paper.recommendations.map(rec => `
                <div class="rec-card">
                    <div class="rec-title">${rec.title}</div>
                    <div class="rec-meta">Sesión: ${rec.session}</div>
                    <div class="rec-score">Similitud: ${rec.similarity_score.toFixed(4)}</div>
                </div>
            `).join('');

            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">${paper.title}</div>
                        <div class="result-rank">Top ${paper.rank}</div>
                    </div>
                    
                    <div class="result-meta">
                        <div class="meta-item">Año: ${paper.year || 'N/A'}</div>
                        <div class="meta-item">Sesión: ${paper.session}</div>
                        <div class="meta-item">Keywords: ${paper.keywords || 'N/A'}</div>
                    </div>
                    
                    <div class="result-abstract">
                        <strong>Abstract:</strong> ${abstractSnippet}
                    </div>
                    
                    <div class="result-scores">
                        <div class="score-badge final">Score Final: ${paper.score_final.toFixed(4)}</div>
                        <div class="score-badge">Títulos: ${paper.score_title.toFixed(4)}</div>
                        <div class="score-badge">Keywords: ${paper.score_keywords.toFixed(4)}</div>
                        <div class="score-badge">Abstract: ${paper.score_abstract.toFixed(4)}</div>
                    </div>
                    <div class="score-formula">Score Final = 0.10 × Título + 0.20 × Keywords + 0.70 × Abstract</div>
                    
                    <div class="recommendations-wrapper">
                        <div class="recommendations-title">Artículos relacionados fuera del Top 10</div>
                        <div class="recommendation-note">Estas recomendaciones se calculan comparando este artículo contra todos los demás documentos, excluyendo el propio artículo y los 10 resultados principales de la búsqueda actual.</div>
                        <div class="recs-grid">
                            ${recsHtml}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        resultsList.innerHTML = html;
        resultsContainer.classList.remove('hidden');
    }

    function renderEmbeddingResults(data) {
        queryDisplay.textContent = data.processed_query + " (por Embeddings)";
        renderQueryInfo(data);
        
        if (data.has_results === false) {
            resultsList.innerHTML = `
                <div class="no-results">
                    <h3>${data.message || "No se encontró información relacionada con tu consulta"}</h3>
                    <p>El dataset contiene artículos científicos de ICMLA 2020 sobre temas de machine learning y aplicaciones relacionadas. Intenta usar términos más cercanos al contenido del dataset.</p>
                    <div class="suggestion-tags">
                        <span class="suggestion-tag">medical image segmentation</span>
                        <span class="suggestion-tag">graph neural networks</span>
                        <span class="suggestion-tag">biometric face recognition</span>
                        <span class="suggestion-tag">reinforcement learning</span>
                        <span class="suggestion-tag">energy forecasting</span>
                        <span class="suggestion-tag">computer vision</span>
                        <span class="suggestion-tag">natural language processing</span>
                    </div>
                </div>
            `;
            resultsContainer.classList.remove('hidden');
            return;
        }

        if (!data.top10 || data.top10.length === 0) {
            resultsList.innerHTML = '<p>No se encontraron resultados.</p>';
            resultsContainer.classList.remove('hidden');
            return;
        }

        const html = data.top10.map(paper => {
            const abstractSnippet = paper.abstract.length > 250 
                ? paper.abstract.substring(0, 250) + '...' 
                : paper.abstract;

            const recsHtml = paper.recommendations.map(rec => `
                <div class="rec-card">
                    <div class="rec-title">${rec.title}</div>
                    <div class="rec-meta">Sesión: ${rec.session}</div>
                    <div class="rec-score">Similitud: ${rec.similarity_score.toFixed(4)}</div>
                </div>
            `).join('');

            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">${paper.title}</div>
                        <div class="result-rank">Top ${paper.rank}</div>
                    </div>
                    
                    <div class="result-meta">
                        <div class="meta-item">Año: ${paper.year || 'N/A'}</div>
                        <div class="meta-item">Sesión: ${paper.session}</div>
                    </div>
                    
                    <div class="result-abstract">
                        <strong>Abstract:</strong> ${abstractSnippet}
                    </div>
                    
                    <div class="result-scores">
                        <div class="score-badge final">Embedding Score: ${paper.embedding_score.toFixed(4)}</div>
                    </div>
                    <div class="score-formula">Este resultado se obtuvo comparando el embedding de la consulta con los embeddings de los abstracts mediante similitud coseno.</div>
                    
                    <div class="recommendations-wrapper">
                        <div class="recommendations-title">Artículos relacionados fuera del Top 10</div>
                        <div class="recommendation-note">Estas recomendaciones se calculan comparando este artículo contra todos los demás documentos mediante embeddings, excluyendo el propio artículo y los 10 resultados principales de la búsqueda actual.</div>
                        <div class="recs-grid">
                            ${recsHtml}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        resultsList.innerHTML = html;
        resultsContainer.classList.remove('hidden');
    }
});
