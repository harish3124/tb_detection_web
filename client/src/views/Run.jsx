export const Run = () => {
  const handleSubmit = (e) => {
    e.preventDefault();
  };
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="audio/*" name="file" required={true} />
        <input type="submit" />
      </form>
    </div>
  );
};
