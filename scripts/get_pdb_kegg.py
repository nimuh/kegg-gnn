

from Bio.KEGG import REST #as KEGG_REST
from Bio.KEGG.KGML.KGML_pathway import Entry
from typing import List, Optional
import graphein.protein as gp
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold, add_k_nn_edges
from graphein.protein.graphs import construct_graph
from graphein.ml.conversion import GraphFormatConvertor
from torch_geometric.utils import subgraph
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT
from typing import List, Optional
import os
from functools import partial



# Standard 20 canonical amino acids and their index mapping (0-19)
# This mapping is crucial for Inverse Folding tasks where the output layer
# has 20 classes corresponding to the residue types.
AA_3_TO_INDEX = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

ko_kegg_db_file = '../data/ko'




def get_pdb_ids_from_kegg(kegg_entry_id: str) -> List[str]:
    """
    Queries the KEGG database for a given entry ID (e.g., 'K00844') and 
    extracts associated PDB identifiers from the DBLINKS section.

    Args:
        kegg_entry_id: The KEGG ID to look up (e.g., K-number for a gene/enzyme).

    Returns:
        A list of PDB IDs (e.g., ['1cdq', '1cdi']) found in the KEGG DBLINKS.
    """
    pdb_ids: List[str] = []
    
    print(f"Querying KEGG for entry: {kegg_entry_id}")

    try:
        # Use KEGG_REST to fetch the entry data
        # We query the 'get' service for the specific ID
        record = REST.kegg_get(kegg_entry_id).read()
        db_info_idx = record.find('DBLINKS')
        db_gene_idx = record.find('GENES')
        print(record[db_info_idx:db_gene_idx])
        
        # KEGG_REST returns a text file format that needs parsing
        if not record:
            print(f"Error: KEGG entry {kegg_entry_id} not found or no data returned.")
            return pdb_ids

        # Use Bio.KEGG.Parser to convert the raw text into a structured object
        # Note: The 'kegg_get' query returns one or more entries; we use a simple loop
        #for entry in Entry.parse(record):
            # The cross-references are typically stored in the 'DBLINKS' dictionary
        #    if hasattr(entry, 'dblink'):
                # DBLINKS is a list of (database_name, ids_string) tuples
        #        for db_name, id_string in entry.dblink:
        #            if db_name == 'PDB':
                        # The id_string is a space-separated string of PDB IDs
        #                pdb_ids.extend(id_string.split())
                        
        print(f"Successfully found {len(pdb_ids)} PDB IDs.")
        
    except Exception as e:
        print(f"An error occurred during KEGG lookup: {e}")
        
    return pdb_ids


def create_protein_graphs(pdb_codes: List[str], 
                          output_dir: str = "pdb_data",
                          graph_format: str = "pyg") -> List:
    """
    Downloads PDB backbone structures, converts them to graphs, and defines 
    the amino acid sequence (residue identity) as the prediction target (data.y).
    
    Args:
        pdb_codes: List of 4-character PDB IDs (e.g., ["3eiy", "4hhb"]).
        output_dir: Directory to store downloaded PDB files.
        graph_format: Output format ("pyg" for PyTorch Geometric, "nx" for NetworkX).
    
    Returns:
        A list of graph objects (PyG Data objects or NetworkX graphs).
    """
    
    # 1. Configuration for Graph Construction
    # We use 'CA' (Alpha Carbon) for backbone-only nodes.
    params_to_change = {
        "granularity": "CA", 
        # Set a reasonable distance threshold and k-NN for local connectivity
        "edge_construction_functions": [
            partial(add_distance_threshold, long_interaction_threshold=8), 
            add_k_nn_edges
        ],
        "edge_construction_function_params": {
            "add_distance_threshold": {"threshold": 8},
            "add_k_nn_edges": {"k": 5} # 5 nearest neighbors
        }
    }
    
    config = ProteinGraphConfig(**params_to_change)

    # 2. Format Converter Setup
    # Extract structural features (coords) for input (data.x)
    # Extract residue name for the target label (data.y)
    # Extract b_factor and meiler as additional optional scalar features (s_feats)
    convertor = GraphFormatConvertor(
        src_format="nx", 
        dst_format=graph_format,
        columns=[
            "coords",       # Node input feature: 3D coordinates (structure)
            "residue_name", # Target feature: 3-letter amino acid code
            "b_factor",     # Optional scalar feature 1
            "meiler",        # Optional scalar feature 2 (Physiochemical properties)
            "edge_index"
        ] 
    )

    graphs = []
    
    print(f"Processing {len(pdb_codes)} proteins...")
    
    for pdb_code in pdb_codes:
        try:
            print(f"Constructing graph for: {pdb_code}")
            
            g = construct_graph(
                config=config, 
                pdb_code=pdb_code,
                #pdb_dir=output_dir
            )
            
            # Convert to PyTorch Geometric Data object
            if graph_format == "pyg":
                data = convertor(g)
                
                # --- Inverse Folding Target Definition ---
                # data.y: The sequence target (index of the amino acid for each residue)
                if hasattr(data, 'residue_name'):
                    # 1. Map 3-letter codes to indices (0-19)
                    targets = [AA_3_TO_INDEX.get(name, -1) for name in data.residue_name]
                    
                    # 2. Filter out residues not in the canonical 20 (e.g., modified or unknown residues)
                    valid_indices = [i for i, target in enumerate(targets) if target != -1]
                    
                    if not valid_indices:
                        print(f"Warning: Skipping {pdb_code} due to no valid canonical residues.")
                        continue


                    # 3. Use the indices for the target (data.y)
                    data.y = torch.tensor([targets[i] for i in valid_indices], dtype=torch.long)
                    
                    # 4. Filter all other tensors to match valid_indices (crucial step for filtering)
                    # This ensures all features and targets have the same number of nodes
                    data.coords = data.coords[valid_indices]
                    data.b_factor = data.b_factor[valid_indices]
                    data.meiler = data.meiler[valid_indices]
                    
                    # --- Inverse Folding Input Feature Definition ---
                    # data.x: The input features (Structure)
                    # For a simple structural input, we use the coordinates.
                    data.x = torch.cat((data.coords, data.meiler), dim=-1) # [num_nodes, 3] tensor
                    data.x = data.x.to(torch.float32)
                    
                    graphs.append(data)
                
        except Exception as e:
            print(f"Failed to process {pdb_code}: {e}")

    return graphs


def read_ko_entries(file):
    ko_ids = []
    with open(file, 'r') as f:
        for line in f:
            ko = line.split()[0]
            ko_ids.append(ko)
    return ko_ids


def main():
    
    
    model = GAT(
        in_channels=10,
        hidden_channels=64,
        num_layers=3,
        out_channels=len(AA_3_TO_INDEX),

    )

    # Subset of PDB IDs to process
    target_pdbs = ["3eiy", 
                   "4hhb", 
                   "1a3n", 
                   "1a2c", 
                   "1tim", 
                   "4hhb",
                   "1rgc",
                   "1r6a",
                   "1zlm",
                   "5n00"] 
    
    # Generate the dataset
    dataset = create_protein_graphs(target_pdbs)
    
    if not dataset:
        print("No graphs were generated.")
        return

    # Create a PyTorch Geometric DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # Simulate a training loop
    for _ in range(100):
        for batch_idx, batch in enumerate(loader):
            pred = model(x=batch.x, edge_index=batch.edge_index)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(loss.item())

if __name__ == "__main__":
    main()