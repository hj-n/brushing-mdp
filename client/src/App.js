import Brushing from "./components/brushing";



function App() {
  const PATH = "http://gpu.hcil.snu.ac.kr:5000/";
  return (
    <div className="App">
      <Brushing 
        url={PATH}
        dataset={"mnist"}
        method={"pca"}
        sample={5}
      />
    </div>
  );
}

export default App;
