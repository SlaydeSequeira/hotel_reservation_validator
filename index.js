const express = require('express');
const axios = require('axios');
const multer = require('multer');
const csvParser = require('csv-parser');
const { MongoClient } = require('mongodb');
const fs = require('fs');

const app = express();
const PORT = 3000;

// MongoDB connection URI and database/collection setup
const MONGO_URI = 'mongodb://localhost:27017'; // Replace with your MongoDB URI
const DATABASE_NAME = 'automatedDatabase';      // New database name
const COLLECTION_NAME = 'uploadedData';         // New collection name

const client = new MongoClient(MONGO_URI);

app.use(express.json());

// Multer setup for file upload
const upload = multer({ dest: 'uploads/' });

// Connect to MongoDB
async function connectToMongo() {
    await client.connect();
    console.log('Connected to MongoDB');
}

// Endpoint to upload CSV and insert data into MongoDB
app.post('/upload-csv', upload.single('file'), async (req, res) => {
    const filePath = req.file.path;
    const collection = client.db(DATABASE_NAME).collection(COLLECTION_NAME);
    
    const results = [];

    fs.createReadStream(filePath)
        .pipe(csvParser())
        .on('data', (data) => results.push(data))
        .on('end', async () => {
            try {
                // Insert data into the collection (MongoDB will create the database and collection if they don't exist)
                await collection.insertMany(results);
                res.status(200).json({ message: 'CSV data inserted into MongoDB successfully' });
            } catch (error) {
                console.error('Error inserting data:', error);
                res.status(500).json({ error: 'Error inserting data into MongoDB' });
            } finally {
                // Delete the file after processing
                fs.unlinkSync(filePath);
            }
        });
});

// Existing endpoint for prediction
app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post('http://localhost:5000/predict', req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error in prediction:', error);
        res.status(500).json({ error: 'Error predicting booking status' });
    }
});

app.listen(PORT, async () => {
    await connectToMongo();
    console.log(`Server running on http://localhost:${PORT}`);
});
