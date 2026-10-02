"""Show and persist final test metrics independently of logging configuration."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def report_final_test_results(config, test_result, *, best_valid_result=None, checkpoint_path=None):
    """Called by rank zero after a successful test evaluation on all ranks."""
    if not test_result:
        raise ValueError('Test evaluation returned no metrics; final results cannot be reported.')
    metrics = {name: float(value) for name, value in test_result.items()}
    payload = {
        'model': config['model'],
        'dataset': config['dataset'],
        'checkpoint': str(checkpoint_path) if checkpoint_path is not None else None,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'best_valid_result': (
            {name: float(value) for name, value in best_valid_result.items()}
            if best_valid_result is not None else None
        ),
        'test_result': metrics,
    }
    document = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    # Keep results visible even when a dependency changes logging handlers or
    # levels. Flush immediately so notebook users see completion without delay.
    print('\nFINAL TEST RESULTS\n' + json.dumps(metrics, indent=2), flush=True)

    destination = Path(config['checkpoint_dir'] or 'saved') / 'test_results.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=destination.parent, prefix='.test_results.', suffix='.tmp', delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(document)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f'Test results saved to: {destination.resolve()}', flush=True)
    return destination
