import React from "react";
function Summary(){
    return (
<div className="container" style={{marginTop: '60px', fontFamily: 'Arial, sans-serif'}}>
    <div className="well2" style={{padding: '20px 40px', backgroundColor: '#f9f9f9', border: '1px solid #e0e0e0'}}>
        <h3 style={{color: '#333', marginBottom: '25px'}}>Our K-Means Model Insights</h3>
        <p>We utilized the k-means algorithm with 4 clusters. The results have been quite insightful.</p>
        <div className="text-center" style={{margin: '25px 0'}}>
            <img 
                src={`${process.env.PUBLIC_URL}/k-means performance.png`} 
                alt="K-Means Performance" 
                className="center-image" 
                style={{width: "800px", borderRadius: '10px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)'}}
            />
        </div>
    </div>

    <div className="well" style={{marginTop: '40px', padding: '20px 40px', backgroundColor: '#f9f9f9', border: '1px solid #e0e0e0'}}>
        <h3 style={{color: '#333', marginBottom: '25px'}}>Customer Clusters</h3>
        <ul className="unstyled">
            <li style={{marginBottom: '10px'}}><strong>Cluster 0:</strong> Customers in this cluster have low cash advance and low one-off purchases</li>
            <li style={{marginBottom: '10px'}}><strong>Cluster 1:</strong> Customers in this cluster have moderate cash advance and high one-off purchases</li>
            <li style={{marginBottom: '10px'}}><strong>Cluster 2:</strong> Customers in this cluster have high cash advance and low one-off purchases</li>
            <li><strong>Cluster 3:</strong> Customers in this cluster have low cash advance and moderate one-off purchases</li>
        </ul>
    </div>
</div>


    
    )
}
export default Summary;