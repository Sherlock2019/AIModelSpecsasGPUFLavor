import { expect, test, type Page } from '@playwright/test'

const shots = process.env.E2E_SCREENSHOT_DIR

async function pickModel(page: Page, query: string, id: string) {
  await page.getByRole('button', { name: 'Model', exact: true }).click()
  await page.getByLabel('Search model').fill(query)
  await page.getByLabel('Search model').press('Enter')
  await expect(page.getByRole('button', { name: 'Model', exact: true })).toHaveAttribute('data-model-id', id)
}

test.beforeEach(async ({ page }) => {
  await page.goto('/guide')
  await page.evaluate(() => localStorage.clear())
})

test('landing page offers the three sizing paths and the AI VM catalog', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'What do you want to size?' })).toBeVisible()
  for (const name of ['Select an LLM', 'Custom LLM', 'Custom ML model']) {
    await expect(page.getByText(name, { exact: true }).first()).toBeVisible()
  }
  for (const f of ['AI-1B', 'AI-8B', 'AI-14B', 'AI-32B', 'AI-70B', 'AI-120B', 'AI-200B', 'AI-400B']) {
    await expect(page.getByRole('heading', { name: f, exact: true })).toBeVisible()
  }
  await expect(page.getByText('Multi-GPU required').first()).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/home.png`, fullPage: true })
})

test('model-first calculation for Llama 3.1 70B', async ({ page }) => {
  await page.goto('/calculator')
  await pickModel(page, 'llama 3.1 70b', 'llama-3.1-70b-instruct')
  await expect(page.getByRole('heading', { name: 'Your GPU requirement' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Matching GPUs' })).toBeVisible()
  await expect(page.getByText(/AI-70B-Q4-/).first()).toBeVisible()
  await expect(page.getByText('AMD', { exact: true }).first()).toBeVisible()
  await expect(page.getByText(/Open it as an AI VM flavor/)).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/calculator.png`, fullPage: true })
  await page.getByRole('button', { name: /Technical details/ }).last().click()
  await page.getByRole('tab', { name: 'Infrastructure output' }).click()
  await expect(page.getByText('PCI_PASSTHROUGH').first()).toBeVisible()
})

test('small model on lowest cost gets a shared vGPU', async ({ page }) => {
  await page.goto('/calculator')
  await pickModel(page, 'qwen3 14b', 'qwen3-14b')
  await page.getByRole('radio', { name: '8K', exact: true }).click()
  await page.getByRole('button', { name: 'Development · 1' }).click()
  await page.getByRole('radio', { name: 'Lowest cost' }).click()
  await expect(page.getByText('AI-14B-Q4-SHARED').first()).toBeVisible()
  await expect(page.getByText('Shared vGPU is enough')).toBeVisible()
})

test('AI flavor: size AI-8B and deploy a spec', async ({ page }) => {
  await page.goto('/flavors/ai-8b')
  await expect(page.getByText('Your AI VM')).toBeVisible()
  await expect(page.getByText('AI-8B-SHARED', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('Shared GPU').first()).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/flavor.png`, fullPage: true })
  await page.getByRole('button', { name: 'Deploy' }).click()
  await expect(page.getByText(/Deployment spec #\d+ created for AI-8B-Q4-SHARED/)).toBeVisible()
  await page.getByRole('tab', { name: 'Compare options' }).click()
  await expect(page.getByRole('cell', { name: 'GPU count' })).toBeVisible()
  await page.goto('/saved')
  await expect(page.getByRole('heading', { name: 'Deployment specs' })).toBeVisible()
  await expect(page.getByText('AI-8B-Q4-SHARED').first()).toBeVisible()
})

test('AI flavor: a bigger model is flagged and can be upgraded', async ({ page }) => {
  await page.goto('/flavors/ai-32b?m=llama-3.3-70b-instruct')
  await expect(page.getByText('This model exceeds the normal AI-32B tier.', { exact: false })).toBeVisible()
  await page.getByRole('button', { name: 'Upgrade to AI-70B' }).click()
  await expect(page).toHaveURL(/\/flavors\/ai-70b/)
  await expect(page.getByText(/^AI-70B-(PRO|MULTI|SHARED)$/).first()).toBeVisible()
  await expect(page.getByText('exceeds the normal')).toHaveCount(0)
})

test('custom ML model: inference, then training shows gradients and optimizer', async ({ page }) => {
  await page.goto('/ml')
  await expect(page.getByText('Custom model analysis')).toBeVisible()
  await expect(page.getByText(/AI-CUSTOM-INF-/).first()).toBeVisible()
  await page.getByRole('button', { name: 'Training', exact: true }).click()
  await page.getByLabel('Parameters (millions)').fill('2000')
  await page.getByRole('button', { name: 'Calculate GPU configuration' }).click()
  await expect(page.getByText(/AI-CUSTOM-TRAIN-/).first()).toBeVisible()
  await expect(page.getByText('Optimizer', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('Estimated activations')).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/ml.png`, fullPage: true })
})

test('catalog and admin pages render', async ({ page }) => {
  for (const [path, heading] of [
    ['/guide', 'How it works'],
    ['/models', 'Model catalog'],
    ['/admin/flavors', 'AI flavor catalog'],
    ['/gpus', 'GPU catalog'],
    ['/compare', 'Compare'],
    ['/benchmarks', 'Benchmarks'],
    ['/saved', 'Saved calculations'],
    ['/settings', 'Settings'],
  ]) {
    await page.goto(path)
    await expect(page.getByRole('heading', { name: heading, exact: true })).toBeVisible()
  }
  await page.goto('/admin/flavors')
  await expect(page.getByText('AI-400B').first()).toBeVisible()
})

test('old shared links still open the calculator', async ({ page }) => {
  await page.goto('/?m=qwen3-8b&p=int4&ctx=8192&n=5')
  await expect(page).toHaveURL(/\/calculator\?/)
  await expect(page.getByRole('button', { name: 'Model', exact: true })).toHaveAttribute('data-model-id', 'qwen3-8b')
})

test('dark theme and mobile layout', async ({ page }) => {
  test.skip(!shots, 'screenshots only')
  await page.goto('/calculator')
  await page.getByRole('radio', { name: 'Dark theme' }).click()
  await expect(page.getByRole('heading', { name: 'Matching GPUs' })).toBeVisible()
  await page.screenshot({ path: `${shots}/dark.png`, fullPage: false })
  await page.getByRole('radio', { name: 'Light theme' }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'What do you want to size?' })).toBeVisible()
  await page.screenshot({ path: `${shots}/mobile.png`, fullPage: true })
})
