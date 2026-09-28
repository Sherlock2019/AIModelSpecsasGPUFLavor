import { expect, test, type Page } from '@playwright/test'

const shots = process.env.E2E_SCREENSHOT_DIR

async function pickModel(page: Page, query: string, id: string) {
  await page.getByRole('button', { name: 'Model', exact: true }).click()
  await page.getByLabel('Search model').fill(query)
  await page.getByLabel('Search model').press('Enter')
  await expect(page.getByRole('button', { name: 'Model', exact: true })).toHaveAttribute('data-model-id', id)
}

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() => localStorage.clear())
  await page.goto('/')
})

test('model-first calculation for Llama 3.1 70B', async ({ page }) => {
  await pickModel(page, 'llama 3.1 70b', 'llama-3.1-70b-instruct')
  await expect(page.getByRole('heading', { name: 'Your GPU requirement' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Matching GPUs' })).toBeVisible()
  await expect(page.getByText(/AI-70B-Q4-/).first()).toBeVisible()
  await expect(page.getByText('AMD', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Why this recommendation?' })).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/calculator.png`, fullPage: true })

  await page.getByRole('button', { name: /Technical details/ }).last().click()
  await page.getByRole('tab', { name: 'Infrastructure output' }).click()
  await expect(page.getByText('pci_passthrough').first()).toBeVisible()
})

test('small model on lowest cost gets a shared vGPU', async ({ page }) => {
  await pickModel(page, 'qwen3 14b', 'qwen3-14b')
  await page.getByRole('radio', { name: '8K', exact: true }).click()
  await page.getByRole('button', { name: 'Development · 1' }).click()
  await page.getByRole('radio', { name: 'Lowest cost' }).click()
  await expect(page.getByText('AI-14B-Q4-SHARED').first()).toBeVisible()
  await expect(page.getByText('Shared vGPU is enough')).toBeVisible()
  if (shots) await page.screenshot({ path: `${shots}/vgpu.png`, fullPage: true })
})

test('model details can be overridden and reset', async ({ page }) => {
  await pickModel(page, 'qwen3 8b', 'qwen3-8b')
  await page.getByRole('button', { name: /Model details/ }).click()
  await page.getByRole('button', { name: 'Edit' }).click()
  await page.getByLabel('Total parameters (B)').fill('9')
  await page.getByRole('button', { name: 'Done' }).click()
  await expect(page.getByText('User override')).toBeVisible()
  await page.getByRole('button', { name: /Reset to official values/ }).click()
  await expect(page.getByText('User override')).toHaveCount(0)
})

test('catalog pages render', async ({ page }) => {
  for (const [path, heading] of [
    ['/guide', 'How it works'],
    ['/models', 'Model catalog'],
    ['/gpus', 'GPU catalog'],
    ['/compare', 'Compare'],
    ['/benchmarks', 'Benchmarks'],
    ['/saved', 'Saved calculations'],
    ['/settings', 'Settings'],
  ]) {
    await page.goto(path)
    await expect(page.getByRole('heading', { name: heading, exact: true })).toBeVisible()
  }
  await page.goto('/compare')
  await page.getByRole('button', { name: /Compare 4 models/ }).click()
  await expect(page.getByRole('cell', { name: 'Llama 3.1 8B Instruct' })).toBeVisible()
})

test('dark theme and mobile layout', async ({ page }) => {
  test.skip(!shots, 'screenshots only')
  await page.getByRole('radio', { name: 'Dark theme' }).click()
  await expect(page.getByRole('heading', { name: 'Matching GPUs' })).toBeVisible()
  await page.screenshot({ path: `${shots}/dark.png`, fullPage: false })
  await page.getByRole('radio', { name: 'Light theme' }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your GPU requirement' })).toBeVisible()
  await page.screenshot({ path: `${shots}/mobile.png`, fullPage: true })
})
