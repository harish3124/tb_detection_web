import { useState } from "react";

export const Run = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null)


    const handleSubmit = (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', file)

        // TODO Change Url for Production
        fetch('http://localhost:5000/api', {
            method: 'POST',
            body: formData,
        }).then((res) => {
            res.json().then((res) => {
                setResult(res.result)
            })
        })
    };
    return (
        <div>
            {(result !== null) && (
                <div>You {result ? '' : 'DONT'} HAVE Tuberculosis</div>
            )}
            <form onSubmit={handleSubmit}>
                <input type="file" accept="audio/*" name="file" required={true} onChange={e => setFile(e.target.files[0])} />
                <input type="submit" />
            </form>
        </div>
    );
};
