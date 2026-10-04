"""Server paths and per-invocation experiment recording (stdlib + PyYAML)."""
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

DATA_ROOT = Path(os.environ.get('CYCLEFLOW_DATA_ROOT', 'datasets'))
EXPS_ROOT = Path(os.environ.get('FLOWCYCLE_EXPS', 'exps'))
os.environ.setdefault('TORCH_HOME', str(EXPS_ROOT / '_cache' / 'torch'))
os.environ.setdefault('MPLCONFIGDIR', str(EXPS_ROOT / '_cache' / 'matplotlib'))
_RUN = None
_PREVIOUS = None


def options():
    import argparse
    import yaml
    p = argparse.ArgumentParser(add_help=False)
    for key in ('config', 'tag', 'variant', 'data_root', 'root', 'data', 'dataset'):
        p.add_argument('--' + key, default=argparse.SUPPRESS)
    args = vars(p.parse_known_args()[0])
    cfg = {}
    if args.get('config'):
        with open(args['config']) as f:
            cfg = yaml.safe_load(f) or {}
    return {**cfg, **args}


def dataset_name(opts):
    if os.environ.get('CYCLEFLOW_DATASET'):
        return os.environ['CYCLEFLOW_DATASET']
    hint = (' '.join(str(opts.get(k, '')) for k in ('data_root', 'root', 'dataset', 'tag'))
            + ' ' + Path(sys.argv[0]).stem).lower()
    if 'mnist' in hint:
        return 'mnist'
    if 'horse2zebra' in hint or 'h2z' in hint:
        return 'horse2zebra'
    if Path(sys.argv[0]).stem == 'make_figures_folder':
        return 'horse2zebra'
    if opts.get('data') == 'folder':
        raise ValueError('Unknown folder dataset: set CYCLEFLOW_DATASET to its name')
    return 'adni'


def component(value):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError(f'Invalid experiment directory name: {value!r}')
    return value


def checkpoint_root():
    return EXPS_ROOT / component(dataset_name(options())) / 'checkpoints'


def resolve_checkpoint(path):
    if path.startswith('exps/checkpoints/'):
        return str(checkpoint_root() / path[len('exps/checkpoints/'):])
    return path


def previous_checkpoint(stage):
    if _PREVIOUS is None:
        raise FileNotFoundError('No previous run for this tag; pass --resume_from explicitly')
    return str(_PREVIOUS / f'stage{stage}.pth')


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.stream.write(value)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


def experiment_root():
    """Allocate immutable run folders; dataset/checkpoints is a latest-run index."""
    global _RUN, _PREVIOUS
    if _RUN is not None:
        return str(_RUN)
    opts = options()
    dataset = component(dataset_name(opts))
    base = EXPS_ROOT / dataset
    if '--help' in sys.argv or '-h' in sys.argv:
        return str(base)
    entry = Path(sys.argv[0]).stem
    check_init = '--check_init' in sys.argv or opts.get('check_init', False)
    training = entry in ('train', 'train_cyclegan', 'train_dit', 'train_revgan') and not check_init
    tag = component(str(opts.get('tag') or (os.environ.get('CYCLEGAN_TAG', 'cyclegan')
                    if entry == 'train_cyclegan' else opts.get('variant', 'morph'))))
    purpose = component(os.environ.get('CYCLEFLOW_PURPOSE',
                        'check_init' if check_init else tag if training else entry))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
    run = base / purpose / stamp
    run.mkdir(parents=True, exist_ok=False)
    index = base / 'checkpoints'
    index.mkdir(exist_ok=True)
    if training:
        target = run / 'checkpoints' / tag
        target.mkdir(parents=True)
        link = index / tag
        if link.is_symlink():
            _PREVIOUS = link.resolve()
        if link.exists() and not link.is_symlink():
            raise FileExistsError(f'Refusing to replace existing checkpoint directory: {link}')
        tmp = index / ('.' + tag + '-' + uuid.uuid4().hex)
        tmp.symlink_to(target, target_is_directory=True)
        tmp.replace(link)
    else:
        # Pin inputs to concrete runs so a concurrent training start cannot
        # change which checkpoint an evaluation reads midway through.
        inputs = run / 'checkpoints'
        inputs.mkdir()
        for path in index.iterdir():
            if not path.name.startswith('.'):
                (inputs / path.name).symlink_to(path.resolve(), target_is_directory=True)
    _RUN = run
    log = open(run / 'console.log', 'a', buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    metadata = dict(argv=sys.argv, options=opts, dataset=dataset, purpose=purpose,
                    python=sys.executable, started_at=stamp, status='running',
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    torch_version=getattr(sys.modules.get('torch'), '__version__', None))
    try:
        metadata['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
        metadata['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=Path(__file__).parent, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    def save():
        (run / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    old_hook = sys.excepthook
    def exception_hook(kind, value, traceback):
        metadata['status'] = 'failed'
        old_hook(kind, value, traceback)
    sys.excepthook = exception_hook
    def finish():
        if metadata['status'] == 'running':
            metadata['status'] = 'exited'
        metadata['ended_at'] = datetime.now(timezone.utc).isoformat()
        save()
        sys.stdout.flush()
        sys.stderr.flush()
    atexit.register(finish)
    save()
    if opts.get('config'):
        (run / 'config.yaml').write_text(Path(opts['config']).read_text())
    print(f'Experiment -> {run}', flush=True)
    return str(run)
