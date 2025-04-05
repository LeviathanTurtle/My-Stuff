const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const url = 'https://uesc.io/';

    //await page.goto(url, { waitUntil: 'networkidle2' });
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 });

    // Save the fully rendered HTML
    const html = await page.content();
    fs.writeFileSync('uesc.html', html);

    // Save all CSS and JS files
    /*
    const resources = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('link[rel="stylesheet"], script[src]'))
            .map(el => el.href || el.src);
    });*/
    const resources = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('link[rel="stylesheet"], script[src]'))
            .map(el => new URL(el.href || el.src, window.location.origin).href);
    });    

    /*const download = async (resourceUrl, folder) => {
        const viewSource = await page.goto(resourceUrl);
        const filePath = path.join(__dirname, folder, path.basename(resourceUrl));
        fs.writeFileSync(filePath, await viewSource.buffer());
    };*/
    /*const download = async (resourceUrl, folder) => {
        try {
            const response = await page.evaluate(async (url) => {
                const res = await fetch(url);
                return res.ok ? await res.arrayBuffer() : null;
            }, resourceUrl);
    
            if (response) {
                const filePath = path.join(__dirname, folder, path.basename(resourceUrl));
                fs.writeFileSync(filePath, Buffer.from(response));
            } else {
                console.log(`Skipping: ${resourceUrl} (CORS issue?)`);
            }
        } catch (e) {
            console.log(`Failed to download: ${resourceUrl} - ${e.message}`);
        }
    };*/
    const download = async (resourceUrl, folder) => {
        try {
            const response = await page.evaluate(async (url) => {
                const res = await fetch(url);
                if (!res.ok) return null;
    
                const buffer = await res.arrayBuffer();
                return Array.from(new Uint8Array(buffer));  // Convert to array for serialization
            }, resourceUrl);
    
            if (response) {
                const buffer = Buffer.from(response);  // Convert back to Buffer
                const fileName = path.basename(resourceUrl.split('?')[0]);  // Remove query params
                const filePath = path.join(__dirname, folder, fileName);
                fs.writeFileSync(filePath, buffer);
            } else {
                console.log(`Skipping (failed to fetch): ${resourceUrl}`);
            }
        } catch (e) {
            console.log(`Failed to download: ${resourceUrl} - ${e.message}`);
        }
    };

    // Create folders and download files
    if (!fs.existsSync('assets')) fs.mkdirSync('assets');
    for (let res of resources) {
        try {
            await download(res, 'assets');
        } catch (e) {
            console.log(`Failed to download: ${res}`);
        }
    }

    await browser.close();
})();
