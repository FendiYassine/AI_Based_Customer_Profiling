import React from 'react';
import './App.css';  // Assuming you put your styles in App.css
import { Link, Route, Routes } from 'react-router-dom';
import NewCustomer from './newCustomer';
import OldCustomer from './oldCustomer';
import Summary from './summary';

function App() {
  return (
    <div className="App">
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
  <div class="container">
    <a class="navbar-brand" href="#">
      <img src={process.env.PUBLIC_URL + '/images/axefinance.png'} alt="" width="85" height="45"/>
    </a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item">
          <Link to='/' class="nav-link active" aria-current="page" href="#">Summary</Link>
        </li>
        <li class="nav-item">
          <Link to='/newCustomer' class="nav-link" >New Customer</Link>
        </li>
        <li class="nav-item">
          <Link to='/oldCustomer' class="nav-link" >Old Customer</Link>
        </li>
        
      </ul>
    </div>
  </div>
</nav>
<Routes>
<Route path='/' element={<Summary/>}></Route>
    <Route path='/newCustomer' element={<NewCustomer/>}></Route>
    <Route path='/oldCustomer' element={<OldCustomer/>}></Route>

</Routes>
  

     
  
    </div>
   
  );
}

export default App;
