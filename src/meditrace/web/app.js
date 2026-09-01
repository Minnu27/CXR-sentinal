const list = document.querySelector('#document-list');
const count = document.querySelector('#document-count');
const form = document.querySelector('#upload-form');
const statusLine = document.querySelector('#form-status');
const fileInput = document.querySelector('#file');

const escapeHtml = (value) => value.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const fileType = name => (name.split('.').pop() || 'DOC').toUpperCase().slice(0, 4);

async function loadDocuments() {
  try {
    const response = await fetch('/api/documents');
    if (!response.ok) throw new Error('Document register is unavailable');
    const data = await response.json();
    count.textContent = String(data.total).padStart(3, '0');
    list.innerHTML = data.items.length ? data.items.map(doc => `
      <div class="document">
        <div class="doc-icon">${fileType(escapeHtml(doc.filename))}</div>
        <div><h3>${escapeHtml(doc.filename)}</h3><p>${escapeHtml(doc.patient_id)} · ${(doc.size_bytes / 1024).toFixed(1)} KB · ${new Date(doc.created_at).toLocaleString()}</p></div>
        <span class="badge">${escapeHtml(doc.status)}</span>
      </div>`).join('') : '<p class="empty">No documents yet. Ingest the first synthetic source to begin.</p>';
  } catch (error) {
    list.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
  }
}

fileInput.addEventListener('change', () => {
  document.querySelector('#file-name').textContent = fileInput.files[0]?.name || 'No source selected';
});
document.querySelector('#refresh').addEventListener('click', loadDocuments);
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  statusLine.textContent = 'Securing source document…';
  try {
    const response = await fetch('/api/documents', {method: 'POST', body: new FormData(form)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Upload failed');
    statusLine.textContent = `Source registered as ${result.id.slice(0, 8)}.`;
    form.reset(); document.querySelector('#file-name').textContent = 'No source selected';
    await loadDocuments();
  } catch (error) { statusLine.textContent = error.message; }
  finally { button.disabled = false; }
});
loadDocuments();
