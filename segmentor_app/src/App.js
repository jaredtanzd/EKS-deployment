import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Footer from './ReportFooter'; // Ensure the correct path if it's in another directory
import './TissueVolumeViewer.css'; // Ensure CSS file is in the same folder or adjust the path accordingly

const App = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const seriesId = "1"; // Replace '1' with the actual series ID you wish to query

    const fetchData = async () => {
      try {
        // Use REACT_APP_BACKEND_URL directly without appending the port
        const tissueUrl = `${process.env.REACT_APP_BACKEND_URL}/inference/tissue/?series=${seriesId}`;
        
        const tissueResponse = await axios.get(tissueUrl);
        setData(tissueResponse.data);
      } catch (error) {
        console.error("Failed to fetch data:", error);
      }
    };
    
    
    
    
    fetchData();
  }, []);

  if (!data) {
    return <div>Loading...</div>;
  }

  return (
    <>
      <div className='pageWrapper'>
        <h2>Patient Information</h2>
        <div className='patientInfo'>
          <p><strong>Name:</strong> {data.patientInfo.patientName}</p>
          <p><strong>ID:</strong> {data.patientInfo.patientID}</p>
          <p><strong>DOB:</strong> {data.patientInfo.patientBirthDate}</p>
          <p><strong>Gender:</strong> {data.patientInfo.patientSex}</p>
          <p><strong>Study Date:</strong> {data.patientInfo.studyDate}</p>
        </div>

        <h2>Image Information</h2>
        <div className='imageInfo'>
          <p><strong>Orientation:</strong> {data.imageInfo.orientation}</p>
          <p><strong>Position:</strong> {data.imageInfo.position}</p>
          <p><strong>Window Center:</strong> {data.imageInfo.windowCenter}</p>
          <p><strong>Window Width:</strong> {data.imageInfo.windowWidth}</p>
        </div>

        <h2>Volume Data</h2>
        <div className='volumeData'>
          {Object.keys(data.volume.total).map((key) => (
            <div key={key} className='volumeSegment'>
              <h3>{key.toUpperCase()}</h3>
              <p>Total Volume: {data.volume.total[key]} ml</p>
              <img src={data.volume.display[key]} alt={`${key} segmentation`} />
            </div>
          ))}
        </div>
      </div>
      <Footer />
    </>
  );
};

export default App;
