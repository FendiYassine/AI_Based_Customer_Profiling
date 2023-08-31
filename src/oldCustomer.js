import React from 'react';

function OldCustomer() {

    function getOldCustomerData() {
        const customerId = document.getElementById('customerId').value;
    
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({cust_id: customerId})
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("Error: " + data.error);
            } else {
                document.getElementById('results').value = "Cluster for customer with ID " + customerId + ": " + data.cluster;
            }
        });
    }
    return(
        <div className='container' style={{marginTop: '40px'}}>
            <div className='row'>
            <div class="input-group mb-3">
  <span class="input-group-text" id="basic-addon1">Customer ID</span>
  <input type="text" class="form-control" id="customerId" placeholder="Customer ID" aria-label="Customer ID" aria-describedby="basic-addon1"/>
</div>

<div class="input-group">
  <span class="input-group-text">Results</span>
  <textarea class="form-control" aria-label="With textarea" readOnly id="results"></textarea>
</div>
        
    <div style={{textAlign: "center",marginTop: '20px'}}>
        <button className='btn btn-success' onClick={getOldCustomerData}>Fetch Data</button>
    </div>
    </div>
  
    </div>
    )
}
export default OldCustomer;