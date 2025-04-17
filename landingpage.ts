import React, { useState } from 'react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [message, setMessage] = useState('');
  const [file, setFile] = useState(null);
  const [generateData, setGenerateData] = useState({ party_a: '', party_b: '', start_date: '', end_date: '', clauses: '' });
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);

  const saveToken = (t) => {
    localStorage.setItem('token', t);
    setToken(t);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken('');
    setResponse(null);
  };

  const handleSocialLogin = (provider) => {
    window.open(`${API_URL}/login/${provider}`, '_blank', 'width=600,height=700');
    alert('After authorizing, copy the token JSON and paste the access_token here.');
  };

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const uploadContract = async () => {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_URL}/upload_contract`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    const data = await res.json();
    setResponse(data);
  };

  const generateContract = async () => {
    const body = {
      ...generateData,
      clauses: generateData.clauses.split(',').map(s => s.trim()),
    };
    const res = await fetch(`${API_URL}/generate_contract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    setResponse(data);
  };

  const callEndpoint = async (path) => {
    const res = await fetch(`${API_URL}/${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ question: query }),
    });
    const data = await res.json();
    setResponse(data);
  };

  if (!token) {
    return (
      <div className="p-8 space-y-4">
        <h1 className="text-2xl font-bold">Login</h1>
        <button onClick={() => handleSocialLogin('instagram')} className="px-4 py-2 bg-pink-500 text-white rounded">Login with Instagram</button>
        <button onClick={() => handleSocialLogin('twitter')} className="px-4 py-2 bg-blue-400 text-white rounded">Login with Twitter</button>
        <button onClick={() => handleSocialLogin('tiktok')} className="px-4 py-2 bg-black text-white rounded">Login with TikTok</button>
        <div className="mt-4">
          <input
            type="text"
            placeholder="Paste access token here"
            value={message}
            onChange={e => setMessage(e.target.value)}
            className="w-full p-2 border rounded"
          />
          <button onClick={() => saveToken(message)} className="mt-2 px-4 py-2 bg-green-500 text-white rounded">Save Token</button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">NDA Manager Dashboard</h1>
        <button onClick={logout} className="px-4 py-2 bg-red-500 text-white rounded">Logout</button>
      </div>

      <div className="border p-4 rounded space-y-4">
        <h2 className="text-xl font-semibold">Upload Contract (.txt)</h2>
        <input type="file" accept=".txt" onChange={handleFileChange} />
        <button onClick={uploadContract} className="px-4 py-2 bg-blue-500 text-white rounded">Upload</button>
      </div>

      <div className="border p-4 rounded space-y-4">
        <h2 className="text-xl font-semibold">Generate Contract</h2>
        <input name="party_a" placeholder="Brand" value={generateData.party_a} onChange={e => setGenerateData({ ...generateData, party_a: e.target.value })} className="w-full p-2 border rounded" />
        <input name="party_b" placeholder="Influencer" value={generateData.party_b} onChange={e => setGenerateData({ ...generateData, party_b: e.target.value })} className="w-full p-2 border rounded" />
        <input type="date" name="start_date" value={generateData.start_date} onChange={e => setGenerateData({ ...generateData, start_date: e.target.value })} className="p-2 border rounded mr-2" />
        <input type="date" name="end_date" value={generateData.end_date} onChange={e => setGenerateData({ ...generateData, end_date: e.target.value })} className="p-2 border rounded" />
        <input placeholder="Clauses (comma separated)" value={generateData.clauses} onChange={e => setGenerateData({ ...generateData, clauses: e.target.value })} className="w-full p-2 border rounded" />
        <button onClick={generateContract} className="px-4 py-2 bg-green-500 text-white rounded">Generate</button>
      </div>

      <div className="border p-4 rounded space-y-4">
        <h2 className="text-xl font-semibold">Query Contract</h2>
        <textarea placeholder="Enter your question" value={query} onChange={e => setQuery(e.target.value)} className="w-full p-2 border rounded" rows={3} />
        <div className="grid grid-cols-3 gap-2">
          {['extract_clauses', 'summarize', 'analyze_risk', 'query'].map(path => (
            <button key={path} onClick={() => callEndpoint(path)} className="px-4 py-2 bg-indigo-500 text-white rounded">{path.replace('_', ' ').toUpperCase()}</button>
          ))}
        </div>
      </div>

      {response && (
        <div className="border p-4 rounded bg-gray-50">
          <h2 className="text-lg font-semibold">Response:</h2>
          <pre className="whitespace-pre-wrap">{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

