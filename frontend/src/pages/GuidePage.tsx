import { ArrowRight, Boxes, Cpu, Gauge, Lightbulb, Share2, SlidersHorizontal } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '../components/ui'
import { toQuery } from '../lib/share'
import { DEFAULT_WORKLOAD } from '../state'
import type { WorkloadInput } from '../types'

const EXAMPLES: { title: string; blurb: string; model: string; patch: Partial<WorkloadInput> }[] = [
  {
    title: 'Llama 3.1 70B for 20 users',
    blurb: 'The classic question: which GPU for a 70B model?',
    model: 'llama-3.1-70b-instruct',
    patch: { precision: 'int4', context_length: 32768, concurrent_sequences: 20 },
  },
  {
    title: 'Small model on a shared GPU',
    blurb: 'Qwen3 14B for a small team, economy priority',
    model: 'qwen3-14b',
    patch: { precision: 'int4', context_length: 8192, concurrent_sequences: 2, average_input_tokens: 2000, average_output_tokens: 500, performance_priority: 'economy' },
  },
  {
    title: 'MoE model: Qwen3 30B-A3B',
    blurb: 'Memory like a 30B, speed like a 3B',
    model: 'qwen3-30b-a3b',
    patch: { precision: 'fp8', context_length: 32768, concurrent_sequences: 32 },
  },
  {
    title: 'Frontier: DeepSeek-R1',
    blurb: '671B MoE at native FP8 across many GPUs',
    model: 'deepseek-r1',
    patch: { precision: 'fp8', context_length: 65536, concurrent_sequences: 16, average_output_tokens: 8000, performance_priority: 'performance' },
  },
]

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface p-5">
      <h2 className="mb-3 flex items-center gap-2 text-base font-semibold text-ink">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-soft text-brand-text">{icon}</span>
        {title}
      </h2>
      <div className="space-y-2 text-sm leading-relaxed text-ink-2">{children}</div>
    </section>
  )
}

function Term({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="py-2">
      <dt className="font-medium text-ink">{name}</dt>
      <dd className="text-sm text-ink-2">{children}</dd>
    </div>
  )
}

export function GuidePage() {
  return (
    <div className="max-w-5xl">
      <PageHeader
        title="How it works"
        description="Tell the calculator what model you want to run and how it will be used. It works out the GPU memory and recommends a vGPU, a dedicated GPU or a multi-GPU setup, plus the matching AI flavor."
      />

      <div className="mb-6 grid gap-3 sm:grid-cols-2">
        {EXAMPLES.map((ex) => (
          <Link
            key={ex.title}
            to={`/calculator?${toQuery(ex.model, { ...DEFAULT_WORKLOAD, ...ex.patch })}`}
            className="group flex items-center justify-between gap-3 rounded-xl border border-line bg-surface p-4 transition hover:border-brand"
          >
            <span>
              <span className="block text-sm font-semibold text-ink">{ex.title}</span>
              <span className="block text-xs text-ink-3">{ex.blurb}</span>
            </span>
            <ArrowRight className="h-4 w-4 text-ink-3 transition group-hover:translate-x-0.5 group-hover:text-brand-text" aria-hidden />
          </Link>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section icon={<SlidersHorizontal className="h-4 w-4" />} title="1. Describe what you want to run">
          <ol className="list-decimal space-y-1.5 pl-5">
            <li>
              <strong className="text-ink">Model:</strong> open the model list and type to search (for example “llama 70b”), or
              pick from Popular. Choose <em>Custom model</em> if yours isn’t listed. <em>Model details</em> shows the official
              values; <em>Edit</em> lets you override them.
            </li>
            <li>
              <strong className="text-ink">Precision:</strong> the size of each weight. The line below shows the weight memory for
              your model. INT4 is the usual serving choice. BF16 gives the highest quality.
            </li>
            <li>
              <strong className="text-ink">Context length:</strong> the longest request (prompt + answer) you need to support.
            </li>
            <li>
              <strong className="text-ink">Concurrent requests:</strong> how many are being answered at the same moment, not total
              users. Use a preset or the − / + buttons.
            </li>
            <li>
              <strong className="text-ink">Performance priority:</strong> Lowest cost favours small or shared GPUs. Max
              performance favours high-end dedicated GPUs.
            </li>
          </ol>
          <p>
            Results update automatically, so there is nothing to submit. Everything else (framework, request lengths, headroom, GPU
            limits) is under <em>Advanced options</em>.
          </p>
        </Section>

        <Section icon={<Gauge className="h-4 w-4" />} title="2. Read the answer">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>
              <strong className="text-ink">Your GPU requirement</strong> describes what <em>any</em> GPU must provide:
              <ul className="mt-1 list-[circle] space-y-1 pl-5">
                <li>memory (weights + context &amp; users + runtime + headroom)</li>
                <li>AI compute and GPU memory speed classes</li>
                <li>how many GPUs, whether a shared GPU is enough, and whether a fast GPU link is needed</li>
              </ul>
            </li>
            <li>
              <strong className="text-ink">Matching GPUs</strong> lists NVIDIA and AMD options ranked for your priority. #1 is the
              recommendation and carries the AI flavor, such as <code>AI-70B-Q4-PROD</code>. Open <em>Technical details</em> on a
              card for specs and the per-GPU calculation.
            </li>
            <li>
              <strong className="text-ink">Memory fit</strong> answers “does it fit?”: green fits, amber is tight, red does not fit.{' '}
              <strong className="text-ink">Performance fit</strong> answers “will it be fast enough?” and is an estimate
              unless benchmark data exists.
            </li>
            <li>
              <strong className="text-ink">Why this recommendation?</strong> shows the memory math and why other GPUs ranked
              lower or were rejected.
            </li>
          </ul>
        </Section>

        <Section icon={<Lightbulb className="h-4 w-4" />} title="3. Explore cheaper options">
          <p>
            <strong className="text-ink">Ways to need less hardware</strong> lists changes that fit the workload on fewer or
            smaller GPUs, such as an FP8 KV cache, lower precision or fewer concurrent requests per instance. Each one lists its
            trade-off. <em>Apply</em> tries it instantly.
          </p>
          <p>
            <strong className="text-ink">Technical details</strong> at the bottom has the memory chart, every formula, the
            OpenStack flavor and <strong className="text-ink">Compare precision</strong> (the same workload at INT4, INT8, FP8 and
            BF16). The{' '}
            <Link to="/compare" className="text-brand-text hover:underline">
              Compare
            </Link>{' '}
            page puts models or GPUs side by side.
          </p>
        </Section>

        <Section icon={<Share2 className="h-4 w-4" />} title="4. Share and hand off">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>
              <strong className="text-ink">Copy link:</strong> the address bar always holds the current calculation.
            </li>
            <li>
              <strong className="text-ink">Export</strong> JSON (with the full calculation trace), YAML or CSV, or download a readable report.
            </li>
            <li>
              <strong className="text-ink">Infrastructure output:</strong> a ready-to-review OpenStack flavor. It is not deployed.
            </li>
            <li>
              <strong className="text-ink">Save</strong> keeps a snapshot under Saved Calculations.
            </li>
          </ul>
        </Section>

        <Section icon={<Boxes className="h-4 w-4" />} title="Good to know">
          <dl className="divide-y divide-line">
            <Term name="KV cache">
              Memory that holds every in-flight token’s attention keys and values. It grows with context × concurrency and can
              rival the model weights.
            </Term>
            <Term name="Dense vs MoE">
              MoE models keep all experts in memory, so memory follows total parameters. Each token only uses a few experts, so
              speed follows active parameters.
            </Term>
            <Term name="vGPU vs dedicated vs multi-GPU">
              A vGPU is a fixed slice of a shared GPU. Dedicated means one whole GPU. Multi-GPU splits one model across several
              GPUs (tensor parallelism).
            </Term>
            <Term name="Safety headroom">
              Extra memory for spikes and fragmentation. The default is 15–20% in production and 10% in development.
            </Term>
          </dl>
        </Section>

        <Section icon={<Cpu className="h-4 w-4" />} title="For administrators">
          <ul className="list-disc space-y-1.5 pl-5">
            <li>
              <Link to="/models" className="text-brand-text hover:underline">
                Models
              </Link>
              : add, edit or disable models, import or export YAML, and set metadata status.
            </li>
            <li>
              <Link to="/gpus" className="text-brand-text hover:underline">
                GPU Catalog
              </Link>
              : set prices (this enables cost ranking), availability, vGPU profiles and site inventory.
            </li>
            <li>
              <Link to="/benchmarks" className="text-brand-text hover:underline">
                Benchmarks
              </Link>
              : add measured results to switch recommendations to benchmark-based.
            </li>
            <li>
              <Link to="/settings" className="text-brand-text hover:underline">
                Settings
              </Link>
              : tune every formula constant, such as overheads, safety margins and priorities.
            </li>
          </ul>
        </Section>
      </div>
    </div>
  )
}
