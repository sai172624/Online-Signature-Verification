const ImageComparison = ({ originalSignature, uploadedSignature }) => (
    <div>
        <h3>Original Signature</h3>
        <img src={`http://localhost:5000${originalSignature}`} alt="Original Signature" />
        
        <h3>Uploaded Signature</h3>
        <img src={`http://localhost:5000${uploadedSignature}`} alt="Uploaded Signature" />
    </div>
);

export default ImageComparison;
