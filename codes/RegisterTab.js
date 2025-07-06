import React, { useState } from 'react';
import { Form, Button, Alert } from 'react-bootstrap';
import axios from 'axios';

const RegisterTab = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!username || !email || !file) {
      setMessage('All fields are required.');
      return;
    }

    const formData = new FormData();
    formData.append('username', username);
    formData.append('email', email);
    formData.append('file', file);

    try {
      await axios.post('http://localhost:5000/register', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage('Registration successful.');
    } catch (error) {
      setMessage('An error occurred.');
    }
  };

  return (
    <div className="p-3">
      {message && <Alert variant="info">{message}</Alert>}
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="username-register">
          <Form.Label>Username</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter username:    "
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group controlId="email-register">
          <Form.Label>Email</Form.Label>
          <Form.Control
            type="email"
        placeholder="Enter email:   "
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group controlId="file-register">
          <Form.Label>Signature File</Form.Label>
          <Form.Control
            type="file"
            onChange={handleFileChange}
            required
          />
        </Form.Group>
        <Button variant="primary" type="submit">
          Register
        </Button>
      </Form>
    </div>
  );
};

export default RegisterTab;
