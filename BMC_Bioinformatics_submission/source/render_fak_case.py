"""Render a source-bound FAK1-P4N case from frozen attention arrays only.

No affinity inference, feature generation, or training is performed. Raw
physical attention is retained without renormalization after masking.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import average_precision_score


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
SOURCE = ROOT / 'runs/manuscript/20261023/final-model-studies/attention/structures/6YOJ_P4N'
BLUE, ORANGE = '#0072B2', '#D55E00'
NAME = 'Figure_6_FAK1_case'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [SOURCE / name for name in ['ibam_matrices.npz', 'ibam_metadata.json',
        'mapping.json', 'contacts.npz', 'validation.json', 'ensemble_bond_map.npz',
        'member_1_bond_map.npz', 'bond_control_reaudit.json']]
    arrays = np.load(sources[0], allow_pickle=False)
    metadata = json.loads(sources[1].read_text())
    mapping = json.loads(sources[2].read_text())
    distance = np.load(sources[3], allow_pickle=False)['distance']
    validation = json.loads(sources[4].read_text())
    bond_export = np.load(sources[5], allow_pickle=False)
    bond_knowledge = np.load(sources[6], allow_pickle=False)['knowledge_mass']
    physical = np.array([kind in ('atom', 'bond') for kind in metadata['token_types']])
    token_ids = np.flatnonzero(physical)
    assert len(token_ids) == 37 and distance.shape == (282, 34)
    l2p = arrays['token_residue'][physical]
    p2l = arrays['residue_token'][:, physical]
    salience = l2p.mean(axis=0)
    resolved = np.isfinite(distance).all(axis=1)
    contacts = resolved & (distance.min(axis=1) <= 4)
    ap = average_precision_score(contacts[resolved], salience[resolved])
    expected = validation['contacts']['4.0']
    assert abs(ap - expected['residue_ap']) < 1e-12
    assert int(resolved.sum()) == expected['resolved_residues'] == 259
    assert int(contacts.sum()) == expected['contact_residues'] == 21
    bond = bond_export['matrix']
    assert bond.shape == (37, 37)
    assert np.array_equal(bond_export['members'][0], bond_export['members'][1])
    assert np.array_equal(bond_export['members'][0], bond_export['members'][2])
    assert np.allclose(bond.sum(axis=1) + bond_knowledge, 1, atol=0.005)

    position = np.arange(1, len(salience) + 1)
    scaled = (salience - salience.min()) / (salience.max() - salience.min())
    rows = mapping['residue_mapping']
    residue_table = pd.DataFrame({
        'construct_position_1based':position,
        'PDB_author_number':[r['auth_seq_id'] if r['auth_seq_id'] not in ('.', '?') else '' for r in rows],
        'PDB_insertion_code':[r['insertion_code'] if r['insertion_code'] not in ('.', '?') else '' for r in rows],
        'residue_name':[r['deposited_residue'] for r in rows],
        'resolved_in_crystal':resolved, 'contact_4A':contacts,
        'minimum_ligand_heavy_atom_distance_A':distance.min(axis=1),
        'physical_ligand_to_protein_mean_attention':salience,
        'display_minmax_attention':scaled,
        'protein_to_ligand_knowledge_token_mass':arrays['residue_token'][:, ~physical].sum(axis=1),
        'knowledge_ligand_to_protein_mean_attention':arrays['token_residue'][~physical].mean(axis=0),
        'protein_to_ligand_knowledge_token_37':arrays['residue_token'][:, 37],
        'protein_to_ligand_knowledge_token_38':arrays['residue_token'][:, 38],
        'ligand_to_protein_knowledge_query_37':arrays['token_residue'][37],
        'ligand_to_protein_knowledge_query_38':arrays['token_residue'][38],
    })
    residue_table.to_csv(OUT / 'tables/fak_case_residues.csv', index=False)
    pd.DataFrame(l2p.T, index=position, columns=[f'bond_token_{i}' for i in token_ids]).rename_axis(
        'construct_position_1based').to_csv(OUT / 'tables/fak_case_ligand_to_protein_iBAM.csv')
    pd.DataFrame(p2l, index=position, columns=[f'bond_token_{i}' for i in token_ids]).rename_axis(
        'construct_position_1based').to_csv(OUT / 'tables/fak_case_protein_to_ligand_iBAM.csv')
    pd.DataFrame(bond, index=token_ids, columns=token_ids).rename_axis('receiving_bond_token').to_csv(
        OUT / 'tables/fak_case_bond_map.csv')
    bond_weights = p2l.mean(axis=0)
    bond_table = pd.DataFrame({
        'physical_token':token_ids,
        'bond_index':[metadata['token_to_bond'][i] for i in token_ids],
        'atom_index_1':[metadata['token_to_atom'][i][0] for i in token_ids],
        'atom_index_2':[metadata['token_to_atom'][i][1] for i in token_ids],
        'PDB_atom_name_1':[mapping['atom_names_in_smiles_order'][metadata['token_to_atom'][i][0]] for i in token_ids],
        'PDB_atom_name_2':[mapping['atom_names_in_smiles_order'][metadata['token_to_atom'][i][1]] for i in token_ids],
        'protein_to_ligand_mean_attention_all_282_residues':bond_weights,
        'ligand_encoder_mean_incoming_physical_attention':bond.mean(axis=0),
        'ligand_encoder_knowledge_token_mass':bond_knowledge,
    })
    bond_table.to_csv(OUT / 'tables/fak_case_bond_weights.csv', index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.2,
        'axes.titlesize':10, 'axes.labelsize':9.2, 'xtick.labelsize':9,
        'ytick.labelsize':9, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'svg.hashsalt':'bindra-fak-case-v1', 'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(7.1, 6.4))
    profile = fig.add_axes([.095, .825, .79, .125])
    heatmap = fig.add_axes([.095, .57, .79, .18])
    bond_ax = fig.add_axes([.095, .075, .315, .35])
    molecule_ax = fig.add_axes([.555, .12, .39, .29])

    # All modeled residues remain visible. Crystal contacts are evaluated only
    # where coordinates exist; missing coordinate ranges are marked in gray.
    unresolved = np.flatnonzero(~resolved)
    for chunk in np.split(unresolved, np.flatnonzero(np.diff(unresolved) > 1) + 1):
        if len(chunk): profile.axvspan(chunk[0]+.5, chunk[-1]+1.5, color='#dddddd', lw=0, zorder=0)
    profile.plot(position, scaled, color=BLUE, lw=.85, zorder=2)
    profile.scatter(position[contacts], scaled[contacts], s=13, color=ORANGE,
        marker='o', label='Crystal contact \u22644 \u00c5', zorder=3)
    profile.set(xlim=(.5, 282.5), ylim=(-.04, 1.10), ylabel='Attention\n(min\u2013max display)',
        title='A  Residue attention and crystal contacts')
    profile.set_yticks([0, .5, 1]); profile.tick_params(labelbottom=False)
    profile.spines[['top', 'right']].set_visible(False)
    profile.legend(loc='upper right', frameon=False, fontsize=9, handletextpad=.3)
    for index, text, location in [(94, 'Cys502', (57, .83)), (124, 'Thr532', (141, .58))]:
        assert rows[index]['auth_seq_id'] == text[3:]
        profile.annotate(text, xy=(position[index], scaled[index]), xytext=location,
            fontsize=9, ha='center', arrowprops={'arrowstyle':'-', 'lw':.6, 'color':'#444444'})

    image = heatmap.imshow(l2p, cmap='viridis', vmin=0, aspect='auto',
        interpolation='nearest', extent=(.5, 282.5, 36.5, -.5), rasterized=True)
    heatmap.set(title='B  Ligand-to-protein iBAM', ylabel='Ligand bond token',
        xlabel='Construct residue (1-based; PDB author mapping exported)', yticks=[0, 12, 24, 36])
    cb = fig.colorbar(image, cax=fig.add_axes([.91, .57, .017, .18]))
    cb.set_label('Attention weight', fontsize=9)
    cb.ax.tick_params(labelsize=9)

    image = bond_ax.imshow(bond, cmap='viridis', vmin=0, aspect='equal',
        interpolation='nearest', rasterized=True)
    bond_ax.set(title='C  Ligand-only bond map', xlabel='Source bond token',
        ylabel='Receiving bond token', xticks=[0, 12, 24, 36], yticks=[0, 12, 24, 36])
    cb = fig.colorbar(image, cax=fig.add_axes([.43, .075, .016, .35]))
    cb.set_label('Attention weight', fontsize=9); cb.ax.tick_params(labelsize=9)

    mol = Chem.MolFromSmiles(mapping['smiles'])
    assert mol.GetNumBonds() == 37 and mol.GetNumAtoms() == 34
    for idx, row in bond_table.iterrows():
        edge = mol.GetBondWithIdx(int(row.bond_index))
        assert {edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()} == {int(row.atom_index_1), int(row.atom_index_2)}
        bond_table.loc[idx, 'atom_element_1'] = mol.GetAtomWithIdx(int(row.atom_index_1)).GetSymbol()
        bond_table.loc[idx, 'atom_element_2'] = mol.GetAtomWithIdx(int(row.atom_index_2)).GetSymbol()
        bond_table.loc[idx, 'bond_type'] = str(edge.GetBondType())
        bond_table.loc[idx, 'bond_is_aromatic'] = bool(edge.GetIsAromatic())
    bond_table.to_csv(OUT / 'tables/fak_case_bond_weights.csv', index=False)
    rdDepictor.Compute2DCoords(mol)
    top_bonds = np.argsort(bond_weights)[-3:]
    norm = Normalize(vmin=0, vmax=float(bond_weights.max()))
    colors = {int(i):tuple(plt.get_cmap('viridis')(norm(weight))[:3]) for i,weight in enumerate(bond_weights)}
    drawer = rdMolDraw2D.MolDraw2DCairo(2100, 1450)
    opts = drawer.drawOptions(); opts.useBWAtomPalette()
    opts.padding = .08; opts.bondLineWidth = 5; opts.highlightBondWidthMultiplier = 5
    opts.fixedFontSize = 96
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=list(range(37)), highlightBondColors=colors)
    drawer.FinishDrawing()
    molecule_image = Image.open(BytesIO(drawer.GetDrawingText())).convert('RGB')
    molecule_ax.imshow(molecule_image);molecule_ax.axis('off')
    label_locations = {0:(.12, .20), 12:(.51, .08), 35:(.82, .83)}
    for idx in top_bonds:
        edge = mol.GetBondWithIdx(int(idx))
        p1, p2 = drawer.GetDrawCoords(edge.GetBeginAtomIdx()), drawer.GetDrawCoords(edge.GetEndAtomIdx())
        midpoint = ((p1.x+p2.x)/2, (p1.y+p2.y)/2)
        molecule_ax.annotate(f'b{idx}', xy=midpoint, xytext=label_locations[int(idx)], textcoords='axes fraction',
            fontsize=9, color='#222222', ha='center', va='bottom',
            bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.88, 'pad':.1},
            arrowprops={'arrowstyle':'-', 'color':'#555555', 'lw':.5})
    fig.text(.555, .455, 'D  Protein-to-ligand bond emphasis', fontsize=9.3, fontweight='normal')
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'),
        cax=fig.add_axes([.555, .075, .39, .018]), orientation='horizontal')
    cb.set_label('Mean attention over 282 modeled residues', fontsize=9)
    cb.ax.tick_params(labelsize=9)

    for ext in ['pdf', 'svg']:
        fig.savefig(OUT / f'figures/{NAME}.{ext}')
    fig.savefig(OUT / f'figures/{NAME}.png', dpi=600)
    with Image.open(OUT / f'figures/{NAME}.png') as raster:
        raster.save(OUT / f'figures/{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    fig.savefig(OUT / f'qa/fak_case_preview.png', dpi=170)
    plt.close(fig)

    # Correct the manuscript supplement's older all-token profile without
    # changing any frozen run export. Its atom view remains an explicitly
    # derived projection of protein-to-ligand bond-token attention.
    derived = np.zeros_like(arrays['residue_atom'])
    for idx, token in enumerate(token_ids):
        ends = metadata['token_to_atom'][int(token)]
        for atom in ends: derived[:, atom] += p2l[:, idx] / len(ends)
    assert np.allclose(derived, arrays['residue_atom'], rtol=2e-5, atol=1e-7)
    derived_ap = average_precision_score((distance[resolved] <= 4).ravel(), derived[resolved].ravel())
    assert abs(derived_ap - expected['pair_p2l_ap']) < 1e-10
    supplement_name = 'Supplementary_Figure_1_FAK_contacts'
    fig = plt.figure(figsize=(7.1, 4.5))
    a = fig.add_axes([.095, .705, .79, .20])
    b = fig.add_axes([.095, .18, .79, .37])
    for chunk in np.split(unresolved, np.flatnonzero(np.diff(unresolved) > 1) + 1):
        if len(chunk):a.axvspan(chunk[0]+.5, chunk[-1]+1.5, color='#dddddd', lw=0)
    a.plot(position, scaled, color=BLUE, lw=.9)
    a.scatter(position[contacts], scaled[contacts], s=13, color=ORANGE, zorder=3,
        label='Crystal contact \u22644 \u00c5')
    a.set(title='A  FAK1-P4N physical-token residue attention', xlim=(.5, 282.5), ylim=(-.04, 1.10),
        ylabel='Attention\n(min\u2013max display)', xlabel='Construct residue (1-based)')
    a.set_yticks([0,.5,1]);a.spines[['top','right']].set_visible(False)
    a.legend(loc='upper right', frameon=False, fontsize=9)
    for index, text, location in [(94, 'Cys502', (57,.83)), (124, 'Thr532', (141,.58))]:
        a.annotate(text, xy=(position[index],scaled[index]), xytext=location,
            fontsize=9, ha='center', arrowprops={'arrowstyle':'-', 'lw':.6, 'color':'#444444'})
    displayed = derived[resolved].T
    image = b.imshow(displayed, interpolation='nearest', aspect='auto', cmap='viridis', vmin=0, rasterized=True)
    iy, ix = np.where((distance[resolved] <= 4).T)
    b.scatter(ix, iy, s=5, facecolors='none', edgecolors='white', linewidths=.35)
    observed_ids = np.flatnonzero(resolved)
    author = np.array([int(rows[i]['auth_seq_id']) for i in observed_ids])
    labels = [450,500,550,600,650]
    ticks = [int(np.argmin(abs(author - value))) for value in labels]
    b.set(title='B  Protein-to-ligand atom projection (derived)',
        ylabel='Ligand atom index', xlabel='Resolved residue (PDB author number)',
        xticks=ticks, xticklabels=[str(author[i]) for i in ticks], yticks=[0,8,16,24,33])
    cb = fig.colorbar(image, cax=fig.add_axes([.90,.18,.016,.37]))
    cb.set_label('Derived attention weight', fontsize=9, labelpad=1);cb.ax.tick_params(labelsize=9, pad=2)
    fig.text(.095,.025,'White circles: atom-residue crystal contacts \u22644 \u00c5. Gray profile regions lack crystal coordinates.',fontsize=9)
    for ext in ['pdf','svg']:
        fig.savefig(OUT / f'figures/{supplement_name}.{ext}')
    fig.savefig(OUT / f'figures/{supplement_name}.png', dpi=600)
    with Image.open(OUT / f'figures/{supplement_name}.png') as raster:
        raster.save(OUT / f'figures/{supplement_name}.tiff', compression='tiff_lzw', dpi=(600,600))
    fig.savefig(OUT / 'qa/fak_case_supplement_preview.png', dpi=170)
    plt.close(fig)
    pd.DataFrame(derived[resolved], index=[str(rows[i]['auth_seq_id']) for i in observed_ids],
        columns=[f'atom_{i}' for i in range(34)]).rename_axis('PDB_author_number').to_csv(
            OUT / 'tables/fak_case_derived_atom_projection.csv')

    share_atom = np.array([[bool(set(metadata['token_to_atom'][i]) & set(metadata['token_to_atom'][j]))
        for j in token_ids] for i in token_ids])
    ranked = np.flatnonzero(resolved)[np.argsort(salience[resolved])[::-1]]
    top_table = residue_table.iloc[ranked[:10]].copy()
    top_table.insert(0, 'attention_rank', np.arange(1, 11))
    top_table.to_csv(OUT / 'tables/fak_case_top10_residues.csv', index=False)
    report = {
        'figure':NAME, 'complex':'FAK1-P4N', 'PDB':'6YOJ', 'resolution_A':mapping['resolution'],
        'ensemble_sha256':validation['ensemble_sha256'], 'prediction_pKd':validation['prediction_pKd'],
        'modeled_construct_residues':282, 'resolved_residues':259, 'contacts_4A':21,
        'physical_bond_tokens':37, 'nonphysical_tokens':2, 'ligand_heavy_atoms':34,
        'residue_AP_4A':ap, 'contact_prevalence':float(contacts.sum()/resolved.sum()),
        'permutation_residue_AP':float(np.mean(expected['permuted_residue_ap'])),
        'top10_contacts':int(contacts[ranked[:10]].sum()), 'top10_enrichment':expected['top10_enrichment'],
        'bond_physical_mass_on_self_diagonal_fraction':float(np.trace(bond)/bond.sum()),
        'bond_physical_mass_between_tokens_sharing_atom_fraction':float(bond[share_atom].sum()/bond.sum()),
        'bond_physical_mass_between_distinct_tokens_sharing_atom_fraction':float(bond[share_atom & ~np.eye(37, dtype=bool)].sum()/bond.sum()),
        'topology_fraction_definition':'Numerator sums physical-to-physical cells whose bond tokens share an endpoint atom; 83.759% includes the 12.363% self-diagonal. The denominator is total physical-to-physical attention mass, excluding knowledge keys.',
        'ligand_encoder_mean_knowledge_token_mass':float(bond_knowledge.mean()),
        'knowledge_mass_definition':'Mean over physical ligand query tokens of outgoing attention allocated to the two nonphysical knowledge keys, not outgoing attention from knowledge-query rows.',
        'protein_to_ligand_mean_knowledge_token_mass_all_residues':float(arrays['nonphysical_mass'].mean()),
        'protein_to_ligand_mean_knowledge_token_mass_resolved_residues':float(arrays['nonphysical_mass'][resolved].mean()),
        'attention_policy':'Last cross-attention layer, mean heads, aligned arithmetic member mean; physical tokens only, no renormalization.',
        'profile_policy':'Mean over 37 physical ligand tokens; min-max scaling over all 282 modeled positions is display-only.',
        'legacy_salience_array_not_used':'The historical residue_salience array aggregates all 39 ligand tokens, including two knowledge tokens. This figure recomputes the mean over only 37 physical tokens, matching the published contact-score implementation and reproducing its AP exactly.',
        'heatmap_policy':'Raw ligand-to-protein attention: 37 physical bond-token queries by all 282 modeled residue keys.',
        'bond_map_policy':'Raw last-layer mean-head physical ligand self-attention; shared frozen encoder produces identical maps across the three members.',
        'molecule_policy':'Raw protein-to-ligand weights averaged over all 282 modeled residues; bond notes identify the three largest means. These are model weights, not causal bond contributions.',
        'ligand_SMILES':mapping['smiles'],
        'molecule_element_font_points_at_native_figure_width':96 * .39 * 7.1 / 2100 * 72,
        'coordinate_policy':'Gray profile regions lack resolved crystal coordinates. Cys502 and Thr532 labels use verified PDB author numbering.',
        'interpretation':'The map highlights both contacting and distant residues. Bond self-attention is target-independent and strongly follows molecular topology. Neither map establishes a binding mechanism.',
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'artifacts':{ext:{'path':f'figures/{NAME}.{ext}', 'sha256':digest(OUT/f'figures/{NAME}.{ext}'),
            'bytes':(OUT/f'figures/{NAME}.{ext}').stat().st_size} for ext in ['pdf', 'svg', 'png', 'tiff']},
        'data_exports':{p.name:digest(p) for p in sorted((OUT/'tables').glob('fak_case_*.csv'))},
        'supplementary_figure':{'name':supplement_name,
            'derived_residue_atom_AP_4A':float(derived_ap),
            'raw_atom_contact_prevalence':float((distance[resolved] <= 4).mean()),
            'projection':'Protein-to-ligand bond-token attention split equally between mapped bond endpoints; no physical-only renormalization.',
            'profile_correction':'Recomputed physical-token mean replaces the historical all-39-token display profile. Frozen run arrays and contact metrics are unchanged.',
            'artifacts':{ext:{'path':f'figures/{supplement_name}.{ext}',
                'sha256':digest(OUT/f'figures/{supplement_name}.{ext}'),
                'bytes':(OUT/f'figures/{supplement_name}.{ext}').stat().st_size} for ext in ['pdf','svg','png','tiff']}},
        'visual_review_status':'pending',
    }
    (OUT / 'qa/fak_case_sources.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['figure','residue_AP_4A','top10_contacts',
        'bond_physical_mass_between_tokens_sharing_atom_fraction','artifacts']}, indent=2))


if __name__ == '__main__':
    main()
