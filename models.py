import ase

def create_calculator(name):
    match name:
        case 'mace-off23-small':
            from mace.calculators.foundations_models import mace_off
            return mace_off('small', default_dtype='float32')
        case 'mace-off24-medium':
            from mace.calculators.foundations_models import mace_off
            return mace_off('https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', default_dtype='float32')
        case 'mace-off23-large':
            from mace.calculators.foundations_models import mace_off
            return mace_off('large', default_dtype='float32')
        case 'mace-omol-0':
            from mace.calculators.foundations_models import mace_omol
            return mace_omol('extra_large', default_dtype='float32')
        case 'mace-mh-1':
            from mace.calculators.foundations_models import mace_mp
            return mace_mp('https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model', default_dtype='float32', head='spice_wB97M')
    raise ValueError(f'Unknown model {name}')

def supports_charge(name):
    return name in ['mace-omol-0']

def supported_elements(name):
    if name.startswith('mace-off'):
        return set(ase.atom.atomic_numbers[symbol] for symbol in ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'])
    if name.startswith('mace-omol') or name.startswith('mace-mh'):
        return set(range(1, 90))
    raise ValueError(f'Unknown model {name}')
