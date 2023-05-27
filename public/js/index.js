function getCard(img, caption)
{
    return `<div class="card" style="width: 18rem;">
            <img src="${img}" class="card-img-top" alt="...">
            <div class="card-body">
                <p class="card-text">${caption}</p>
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
}, 300));

