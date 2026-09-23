// Optional browser verification: npm install --no-save playwright && node tests/ui_smoke.cjs
// Uses only presets and the A* baseline, so it makes no model API calls.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const base = process.env.MAPF_TEST_URL || 'http://127.0.0.1:8091';
const shot = name => path.join(__dirname, '..', 'outputs', name);

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(base);
    await page.waitForSelector('html[data-ready="true"]');

    // Presets: open the first, scrub, check decision cards follow the timeline.
    const presets = await page.$$('.preset');
    assert.ok(presets.length > 0, 'bundled presets are listed');
    await presets[0].click();
    await page.waitForFunction(() => document.getElementById('run-title').textContent.includes('REPLAY'));
    await page.$eval('#timeline', el => { el.value = 2; el.dispatchEvent(new Event('input')); });
    await page.waitForFunction(() => document.getElementById('tick-label').textContent === 'T2');
    assert.ok((await page.$$('.agent')).length > 0, 'agent decision cards render');
    await page.screenshot({path: shot('ui-preset.png')});

    // Live A* run on an official maze.
    await page.click('#live');
    await page.click('#models button[data-provider="astar"]');
    await page.selectOption('#case', 'validation-mazes-seed-000');
    await page.fill('#max-steps', '40');
    await page.click('#reset');
    await page.waitForFunction(() => document.getElementById('steps').textContent === '0');
    await page.click('#run');
    await page.waitForFunction(() => document.getElementById('status').textContent === 'SOLVED', null, {timeout: 60000});
    assert.equal(await page.textContent('#steps'), '23');
    await page.$eval('#timeline', el => { el.value = 10; el.dispatchEvent(new Event('input')); });
    await page.waitForFunction(() => document.getElementById('tick-label').textContent === 'T10');
    await page.screenshot({path: shot('ui-live.png')});

    const mobile = await browser.newPage({viewport: {width: 390, height: 844}});
    await mobile.goto(base); await mobile.waitForSelector('html[data-ready="true"]');
    assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no horizontal overflow');
    await mobile.screenshot({path: shot('ui-mobile.png')});
    assert.deepEqual(errors, []);
    console.log('ui smoke ok');
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
