export async function getSummary(text) {
    const response = await fetch('http://localhost:5000/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });
    const data = await response.json();
    return data.summary;
}
