import React, { useState } from 'react';
import { getSummary } from '../services/api';

const Summarizer = () => {
    const [inputText, setInputText] = useState('');
    const [summary, setSummary] = useState('');

    const handleSummarize = async () => {
        const result = await getSummary(inputText);
        setSummary(result);
    };

    return (
        <div className="container">
            <textarea
                className="input-box"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to summarize"
            />
            <button onClick={handleSummarize} className="btn">
                Summarize
            </button>
            <textarea
                className="output-box"
                value={summary}
                readOnly
                placeholder="Summary will appear here"
            />
        </div>
    );
};

export default Summarizer;
