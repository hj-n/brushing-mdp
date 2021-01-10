import React, { useEffect } from 'react';
import axios from 'axios';
import * as d3 from 'd3';

function initializeScatterplotD3(coor, density, id, margin, gSize, xScale, yScale, opacityScale, url) {
    // Add Grouping for the scatterplot
    const radius = 2.7;
    const color = "black";
    const injection = new Array(density.length).fill(0);
    
    
    // For lens
    const lensSvg = d3.select("#" + id).append("g");


    // For scatterplot
    const svg = d3.select("#" + id)
                  .append("g")
                  .attr("id", id + "-g")
                  .attr("transform", "translate(" + margin + "," + margin + ")");

    const circle = svg.selectAll("circle")
                      .data(injection)
                      .enter()
                      .append("circle")
                      .attr("r", radius)
                      .attr("fill", color)
                      .attr("cx", (_, i) => xScale(coor[i][0]))
                      .attr("cy", (_, i) => yScale(coor[i][1]))
                      .style("opacity", (_, i) => opacityScale(density[i]));
    
    circle.on("mouseover", function(e, d) {
           const nodes = circle.nodes();
           const i = nodes.indexOf(this);
           axios.get(url + "/similarity", {
               params: {index: i}
           }).then(response => {
               const max = response.data.max;
               const similarity = response.data.similarity;
               
               circle.data(similarity)
                     .join(
                         enter => {},
                         update => {
                             update.attr("r", (d, idx) => { 
                                        if (d > 0 && i !== idx) return radius * 1.5;
                                        else if (i === idx) return radius * 2.5;
                                        else return radius;
                                   })
                                   .attr("fill", (d, idx) => { 
                                        if (d > 0 && i !== idx) return "red";
                                        else return "black";
                                   })
                                   .style("opacity", (d, idx) => { 
                                        if (d > 0 && i !== idx) return d / max;
                                        else return opacityScale(density[idx]);
                                   });
                         }
                     );
           });
       })
       .on("mouseout", function() {
           circle.data(injection)
                 .attr("r", radius)
                 .attr("fill", color)
                 .style("opacity", (_, i) => opacityScale(density[i]));

       })


}


const Brushing = (props) => {


    // Global variables for the hook
    const size = 740;
    const margin = 20;
    const gSize = size - margin * 2;
    let xScale;
    let yScale;
    let opacityScale;

    useEffect(async () => {
        const url = props.url;
        const params = {
            dataset: props.dataset,
            method:  props.method,
            sample: props.sample,
        }
        const result = await axios.get(url + "init", { params: params });
        if (result.status === 400) { alert('No such dataset exists!!'); return; }

        const basicInfo = await axios.get(url + "basic", { parmas : params });
        console.log(basicInfo);

        let coor = basicInfo.data.emb;
        let density = basicInfo.data.density;

        let xCoor = coor.map(d => d[0]);
        let yCoor = coor.map(d => d[1]);

        const xDomain = [d3.min(xCoor), d3.max(xCoor)];
        const yDomain = [d3.min(yCoor), d3.max(yCoor)];
        const xDomainSize = xDomain[1] - xDomain[0];
        const yDomainSize = yDomain[1] - yDomain[0];

        let xRange, yRange;
        if (xDomainSize > yDomainSize) {
            let difference = gSize * ((xDomainSize - yDomainSize) / xDomainSize);
            xRange = [0, gSize];
            yRange = [difference, gSize -difference];
        }
        else {
            let difference = gSize * ((yDomainSize - xDomainSize) / xDomainSize);
            xRange = [difference, gSize -difference];
            yRange = [0, gSize];
        }


        xScale = d3.scaleLinear().domain(xDomain).range(xRange);
        yScale = d3.scaleLinear().domain(yDomain).range(yRange);
        opacityScale = d3.scaleLinear().domain([d3.min(density),d3.max(density)]).range([0.1, 1]);

        console.log(d3.min(xCoor) - d3.max(xCoor))
        console.log(d3.min(yCoor) - d3.max(yCoor))

        initializeScatterplotD3(coor, density, "d3-brushing", margin, gSize, xScale, yScale, opacityScale, url);

    }, []);



    return (
        <div style={{margin: "auto", width: size}}>
            <svg id="d3-brushing" 
                 width={size}
                 height={size}
                 style={{
                     marginTop: 30,
                     border: "1px solid black",
                 }}
            ></svg>
        </div>
    );
}

export default Brushing;