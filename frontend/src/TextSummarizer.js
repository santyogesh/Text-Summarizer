import React, { useState } from 'react';
import axios from 'axios';

const TextSummarizer = () => {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [length, setLength] = useState(0);  // Optional summary length

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/summarize', {
        text: inputText,
        length: length,
      });
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error summarizing text', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-semibold text-center mb-4">Text Summarizer</h1>
      
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto bg-white p-4 rounded-lg shadow-lg space-y-4">
        <div className="space-y-2">
          <textarea
            value={inputText}
            onChange={handleInputChange}
            rows="10"
            className="w-full p-2 border border-gray-300 rounded-md"
            placeholder="Enter your text here"
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-lg">Summary Length (Optional)</label>
          <input
            type="number"
            value={length}
            onChange={(e) => setLength(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md"
            placeholder="e.g., 100" 
          />
        </div>

        <button
          type="submit"
          className="w-full p-2 bg-blue-500 text-white rounded-md"
        >
          {loading ? 'Summarizing...' : 'Get Summary'}
        </button>
      </form>

      {summary && (
        <div className="mt-6 bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-2">Summary</h2>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
};

export default TextSummarizer;
