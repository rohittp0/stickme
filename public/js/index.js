function getCard(img, caption)
{
    const p = document.createElement("p");
    p.innerText = caption;

    return `<div class="card" style="width: 18rem;">
            <img src="${img}" class="card-img-top" alt="${p.innerText}" id="${img}">
            <div class="card-body">
                <p class="card-text">${p.innerText}</p>
            </div>
            <div class="card-footer">
                <button class="btn btn-primary" onclick="share('${img}', \`${p.innerText}\`)">Share</button>
            </div>
        </div>`;
}


function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

document.getElementById("search").addEventListener("keyup", debounce((e) => {
    const search = e.target.value;
    const results = document.getElementById("results");
    results.innerHTML = "";

    if (search.length > 0) {
        fetch(`/search?q=${search}`)
            .then(response => response.json())
            .then(data => {
                data.forEach(element => {
                    results.innerHTML += getCard(element.img, element.caption);
                });
            });
    }
}, 1000));


const share = async (id, caption) => {
    const img = document.getElementById(id);
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    canvas.toBlob(function(blob) {
        if (navigator.share) {
            navigator.share({
                files: [new File([blob], 'image.png', {type: blob.type})],
                title: 'StickMe',
                text: caption
            })
                .then(() => console.log('Successful share'))
                .catch((error) => console.log('Error sharing', error));
        }
    });

};

