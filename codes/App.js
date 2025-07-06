import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import ImageComparison from './ImageComparison';


const App = () => {
  const [activeTab, setActiveTab] = useState('register');
  const [registerForm, setRegisterForm] = useState({ username: '', email: '', password: '', file: null });
  const [verifyForm, setVerifyForm] = useState({ username: '', password: '', file: null });
  const [verificationResult, setVerificationResult] = useState('');
  const [similarityIndex, setSimilarityIndex] = useState(null);
  const [originalSignature, setOriginalSignature] = useState(null);
  const [uploadedSignature, setUploadedSignature] = useState(null);

  // Switch between tabs
  const handleTabClick = (tab) => {
    setActiveTab(tab);
  };

  // Handle changes in the register form
  const handleRegisterChange = (e) => {
    const { name, value, files } = e.target;
    setRegisterForm((prevForm) => ({ ...prevForm, [name]: files ? files[0] : value }));
  };

  // Handle changes in the verify form
  const handleVerifyChange = (e) => {
    const { name, value, files } = e.target;
    setVerifyForm((prevForm) => ({ ...prevForm, [name]: files ? files[0] : value }));
  };

  // Handle registration form submission
  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('username', registerForm.username);
    formData.append('email', registerForm.email);
    formData.append('password', registerForm.password);
    formData.append('file', registerForm.file);

    axios.post('http://localhost:5000/register', formData)
      .then(response => {
        alert(response.data.message);
      })
      .catch(error => {
        if (error.response && error.response.data) {
          alert(error.response.data.error);
        } else {
          alert('An error occurred during registration.');
        }
      });
  };

  // Handle verification form submission
  const handleVerifySubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('username', verifyForm.username);
    formData.append('password', verifyForm.password);
    formData.append('file', verifyForm.file);

    axios.post('http://localhost:5000/verify', formData)
      .then(response => {
        console.log('Original Signature Path:', response.data.org);
        console.log('Original Signature Path:', response.data.uploaded);
        
        setVerificationResult(response.data.prediction);
        setSimilarityIndex(response.data.simind)
        setOriginalSignature(response.data.org)
        setUploadedSignature(response.data.uploaded)
      })
      .catch(error => {
        if (error.response && error.response.data) {
          alert(error.response.data.error);
        } else {
          alert('An error occurred during verification.');
        }
      });  
  };


  return (
    <div className="container">
      <h1 className="title">Signature Verification System</h1>
      <ul className="nav-tabs">
        <li
          className={`nav-tab ${activeTab === 'register' ? 'active' : ''}`}
          onClick={() => handleTabClick('register')}
        >
          Register
        </li>
        <li
          className={`nav-tab ${activeTab === 'verify' ? 'active' : ''}`}
          onClick={() => handleTabClick('verify')}
        >
          Verify
        </li>
      </ul>
      <div className="tab-content">
        {activeTab === 'register' && (
          <div className="tab-pane active">
            <form onSubmit={handleRegisterSubmit}>
              <div className="form-group">
                <label htmlFor="username-register">Username</label>
                <input
                  type="text"
                  id="username-register"
                  name="username"
                  value={registerForm.username}
                  onChange={handleRegisterChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="email-register">Email</label>
                <input
                  type="email"
                  id="email-register"
                  name="email"
                  value={registerForm.email}
                  onChange={handleRegisterChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="password-register">Password</label>
                <input
                  type="password"
                  id="password-register"
                  name="password"
                  value={registerForm.password}
                  onChange={handleRegisterChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="file-register">Upload Signature Image</label>
                <input
                  type="file"
                  id="file-register"
                  name="file"
                  onChange={handleRegisterChange}
                  required
                />
              </div>
              <button type="submit">Register</button>
            </form>
          </div>
        )}
        {activeTab === 'verify' && (
          <div className="tab-pane active">
            <form onSubmit={handleVerifySubmit}>
              <div className="form-group">
                <label htmlFor="username-verify">Username</label>
                <input
                  type="text"
                  id="username-verify"
                  name="username"
                  value={verifyForm.username}
                  onChange={handleVerifyChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="password-verify">Password</label>
                <input
                  type="password"
                  id="password-verify"
                  name="password"
                  value={verifyForm.password}
                  onChange={handleVerifyChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="file-verify">Upload Image for Verification</label>
                <input
                  type="file"
                  id="file-verify"
                  name="file"
                  onChange={handleVerifyChange}
                />
              </div>
              <button type="submit">Verify</button>
            </form>
                  <p className="result">Verification Result: {verificationResult}</p>
                  <p className="result">Euclidean Distance: {similarityIndex}</p>
                
              <div className="image-comparison">
                <h3>Original Signature</h3>
                <img src= {originalSignature} alt="Original Signature" />
                <h3>Uploaded Signature</h3>
                <img src= {uploadedSignature} alt="Uploaded Signature" />
              </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
