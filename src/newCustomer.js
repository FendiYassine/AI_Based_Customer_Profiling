import React from 'react';

function NewCustomer() {
    const handleSaveToCSV = () => {
        const tableRows = document.querySelectorAll('#newCustomer tbody tr');
        const data = Array.from(tableRows).map(row => {
            const inputs = row.querySelectorAll('input');
            return Array.from(inputs).map(input => input.value);
        });
    
        // Convert data to a format suitable for sending to server
        const columns = ['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
                         'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                         'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                         'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
                         'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT','TENURE'];
        const payload = data.map(row => {
            const obj = {};
            row.forEach((value, index) => {
                obj[columns[index]] = value;
            });
            return obj;
        });
    
        // Send data to Flask backend
        fetch("http://localhost:5000/save-csv", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload[0]),  // send the first row for simplicity
        })
        .then(response => response.json())
        .then(data => {
            // Set the received data into the textarea
            document.getElementById('results1').value = "Cluster for customer with ID " + data.cust_id + ": " + data.cluster;
            console.log(data.message);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }
    return(
        <div className='container' style={{marginTop: '40px'}}>
            <div className='row'>
        <table className='table-bordered' id="newCustomer">
          <thead>
              <tr>
                  <th>CUST_ID</th>
                  <th>BALANCE</th>
                  <th>BALANCE_FREQUENCY</th>
                  <th>PURCHASES</th>
                  <th>ONEOFF_PURCHASES</th>
                  <th>INSTALLMENTS_PURCHASES</th>
                  <th>CASH_ADVANCE</th>
                  <th>PURCHASES_FREQUENCY</th>
                  <th>ONEOFF_PURCHASES_FREQUENCY</th>
                  <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
                  <th>CASH_ADVANCE_FREQUENCY</th>
                  <th>CASH_ADVANCE_TRX</th>
                  <th>PURCHASES_TRX</th>
                  <th>CREDIT_LIMIT</th>
                  <th>TENURE</th>
              </tr>
          </thead>
          <tbody>
              <tr>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
                  <td><input type="text" className='form-control'/></td>
              </tr>
          </tbody>
      </table>
      </div>
      <div className='row'>
      <div style={{textAlign: "center",marginTop: '20px'}}>
      <button className='btn btn-success' onClick={handleSaveToCSV}>Submit</button>
      </div>
      <div style={{textAlign: "center",marginTop: '20px'}}>
      <div class="input-group">
  <span class="input-group-text">Results</span>
  <textarea class="form-control" aria-label="With textarea" readOnly id="results1"></textarea>
</div>
        </div>
        </div>
  </div>
      
    )
}
export default NewCustomer;