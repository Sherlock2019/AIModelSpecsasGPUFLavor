import { ArrowRight, Boxes, BrainCircuit, PenLine, ScanEye } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link, Navigate, useLocation } from 'react-router-dom'
import { api } from '../api'
import { Alert, Badge, Spinner } from '../components/ui'
import { useCalc } from '../state'
import type { AIFlavor } from '../types'

const GROUPS: { id: AIFlavor['group']; label: string; blurb: string }[] = [
  { id: 'small', label: 'Small', blurb: 'Assistants, chatbots and extraction' },
  { id: 'medium', label: 'Medium', blurb: 'RAG, coding and reasoning' },
  { id: 'large', label: 'Large', blurb: 'Enterprise-grade assistants' },
  { id: 'ultra', label: 'Ultra', blurb: 'Frontier-scale models' },
]

function ChoiceCard({ to, icon, title, blurb, cta }: { to: string; icon: ReactNode; title: string; blurb: string; cta: string }) {
  return (
    <Link
      to={to}
      className="group flex flex-col rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)] transition hover:-translate-y-0.5 hover:border-brand hover:shadow-md"
    >
      <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-brand-soft text-brand-text">{icon}</span>
      <span className="text-base font-semibold tracking-wide text-ink uppercase">{title}</span>
      <span className="mt-1 flex-1 text-sm text-ink-2">{blurb}</span>
      <span className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-brand-text">
        {cta} <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" aria-hidden />
      </span>
    </Link>
  )
}

export function FlavorCard({ flavor, modelNames }: { flavor: AIFlavor; modelNames: Map<string, string> }) {
  const examples = flavor.example_model_ids.map((id) => modelNames.get(id)).filter(Boolean).slice(0, 3) as string[]
  const multiOnly = !flavor.allow_vgpu && !flavor.allow_full_gpu
  return (
    <article className="flex flex-col rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-mono text-lg font-semibold text-ink">{flavor.display_name}</h3>
          <p className="text-sm text-ink-2">{flavor.description}</p>
        </div>
        {multiOnly && <Badge tone="warning">Multi-GPU required</Badge>}
      </div>
      <p className="mt-2 text-xs text-ink-3">
        Up to ~{flavor.parameter_ceiling_b}B model class · offered as{' '}
        {[flavor.allow_vgpu && 'shared', flavor.allow_full_gpu && 'dedicated', flavor.allow_multi_gpu && 'multi-GPU'].filter(Boolean).join(' / ')}
      </p>
      <dl className="mt-4 space-y-3 text-sm">
        <div>
          <dt className="text-xs text-ink-3">Examples</dt>
          <dd className="text-ink">{examples.length ? examples.join(' · ') : '—'}</dd>
        </div>
        <div>
          <dt className="text-xs text-ink-3">Typical GPU memory</dt>
          <dd className="font-semibold text-ink">
            {flavor.default_vram_min_gb}–{flavor.default_vram_max_gb}
            {flavor.group === 'ultra' || flavor.group === 'large' ? '+' : ''} GB
          </dd>
        </div>
        {flavor.recommended_use_cases.length > 0 && (
          <div>
            <dt className="text-xs text-ink-3">Best for</dt>
            <dd className="text-ink-2">{flavor.recommended_use_cases.join(' · ')}</dd>
          </div>
        )}
      </dl>
      <Link
        to={`/flavors/${flavor.id}`}
        className="mt-5 inline-flex h-9 items-center justify-center rounded-lg bg-brand px-4 text-sm font-medium text-brand-ink hover:brightness-110"
      >
        Configure
      </Link>
    </article>
  )
}

export function HomePage() {
  const location = useLocation()
  const { models } = useCalc()
  const [flavors, setFlavors] = useState<AIFlavor[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.aiFlavors().then(setFlavors).catch((e: Error) => setError(e.message))
  }, [])

  // Old shared calculator links (/?m=...) now live under /calculator.
  if (new URLSearchParams(location.search).has('m')) return <Navigate to={`/calculator${location.search}`} replace />

  const names = new Map(models.map((m) => [m.id, m.name.replace(/ Instruct$/, '')]))
  return (
    <div className="mx-auto max-w-[1400px] space-y-10">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-ink">What do you want to size?</h1>
        <p className="mt-1 text-sm text-ink-2">Describe the AI workload. We’ll work out the GPU: shared vGPU, dedicated GPU or multi-GPU, NVIDIA or AMD.</p>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <ChoiceCard to="/calculator" icon={<Boxes className="h-5 w-5" />} title="Select an LLM" blurb="Llama, Qwen, DeepSeek, Mistral, Gemma, Phi and more from the catalog." cta="Choose a model" />
          <ChoiceCard to="/calculator?custom=1" icon={<PenLine className="h-5 w-5" />} title="Custom LLM" blurb="Your own language model: enter its parameters and architecture." cta="Enter parameters" />
          <ChoiceCard to="/ml" icon={<ScanEye className="h-5 w-5" />} title="Custom ML model" blurb="Vision, speech, diffusion, embeddings, tabular or any network — inference, training or fine-tuning." cta="Size an ML model" />
        </div>
      </section>

      <section aria-labelledby="flavors-title">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 id="flavors-title" className="text-xl font-semibold text-ink">
              Or start from an AI VM flavor
            </h2>
            <p className="text-sm text-ink-2">Pick an AI capability. The calculator decides the infrastructure behind it.</p>
          </div>
        </div>
        {error && <Alert tone="critical">{error}</Alert>}
        {!flavors && !error && (
          <div className="flex items-center gap-2 text-sm text-ink-3">
            <Spinner /> Loading flavors…
          </div>
        )}
        {flavors && (
          <div className="space-y-8">
            {GROUPS.map((g) => {
              const list = flavors.filter((f) => f.group === g.id && f.enabled)
              if (!list.length) return null
              return (
                <div key={g.id}>
                  <div className="mb-3 flex items-baseline gap-3">
                    <h3 className="text-xs font-semibold tracking-wider text-ink-3 uppercase">{g.label}</h3>
                    <span className="text-xs text-ink-3">{g.blurb}</span>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {list.map((f) => (
                      <FlavorCard key={f.id} flavor={f} modelNames={names} />
                    ))}
                  </div>
                </div>
              )
            })}
            <div>
              <div className="mb-3 flex items-baseline gap-3">
                <h3 className="text-xs font-semibold tracking-wider text-ink-3 uppercase">Custom</h3>
                <span className="text-xs text-ink-3">Anything not in the catalog</span>
              </div>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                <ChoiceCard to="/calculator?custom=1" icon={<PenLine className="h-5 w-5" />} title="Custom LLM" blurb="Enter parameters; the calculator assigns the AI flavor tier." cta="Configure" />
                <ChoiceCard to="/ml" icon={<BrainCircuit className="h-5 w-5" />} title="Custom ML model" blurb="Generates an AI-CUSTOM flavor such as AI-CUSTOM-INF-24G." cta="Configure" />
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
