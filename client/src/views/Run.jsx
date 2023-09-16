export const Run = () => {
    const handleSubmit = (e) => {
        e.preventDefault();
    }
    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" accept="audio/*" required={true} />
                <input type="submit" />
            </form>
        </div>
    )
}
