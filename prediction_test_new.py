import pandas as pd
from typing import Any, Dict,List, Optional,Tuple
from tqdm import tqdm
import math
import os
import shutil
import duckdb
import numpy as np
import torch
import torch_frame
import torch_geometric
import torch.nn.functional as F
import relbench
#from relbench.modeling.utils import infer_df_stype
from accelerate.commands.config.config_args import cache_dir
from relbench.modeling.utils import get_stype_proposal
from sklearn.conftest import print_changed_only_false
from torch_geometric.data import Data, Database,HeteroData
from torch_geometric.nn import GCNConv
from relbench.datasets import get_dataset,get_dataset_names, register_dataset
from relbench.tasks import get_task,get_task_names
from relbench.datasets import Dataset
from relbench.base import Database,Dataset,Table,EntityTask,TaskType,RecommendationTask
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
import relbench.modeling.graph
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.graph import get_node_train_table_input
from torch_geometric.loader import NeighborLoader
from sentence_transformers import SentenceTransformer
from torch.nn import BCEWithLogitsLoss,Embedding, ModuleDict,L1Loss
from relbench.metrics import accuracy, average_precision, f1, roc_auc
import copy
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.seed import seed_everything



if os.path.exists("./cache/my_dataset"):
    shutil.rmtree("./cache/my_dataset")


class Group(Dataset):

    val_timestamp = pd.Timestamp("2018-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def make_db(self)->Database:
        azienda= pd.read_csv(filepath_or_buffer="C:\\Users\\Marco\\Desktop\\tesi\\file_csv\\companies_final.csv", sep=",")
        gruppo = pd.read_csv(filepath_or_buffer="C:\\Users\\Marco\\Desktop\\tesi\\file_csv\\groups_final.csv", sep=",")
        relazioni_societarie = pd.read_csv(filepath_or_buffer="C:\\Users\\Marco\\Desktop\\tesi\\file_csv\\membership_final.csv", sep=",")

        #relazioni_societarie["date_from"]=pd.to_datetime(relazioni_societarie["date_from"],dayfirst=True)
        #relazioni_societarie["date_to"] = pd.to_datetime(relazioni_societarie["date_to"],dayfirst=True)
        relazioni_societarie['id'] = range(len(relazioni_societarie))

        for df in (azienda, gruppo, relazioni_societarie):
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].str.match(r'^\d{1,2}/\d{1,2}/\d{4}').any():
                    df[col] = pd.to_datetime(df[col], dayfirst=True)






        tables={}

        tables["azienda"] = Table(df=pd.DataFrame(azienda), pkey_col="company_id",
                                            fkey_col_to_pkey_table={},time_col=None)

        tables["gruppo"] = Table(df=pd.DataFrame(gruppo),pkey_col="group_id",
                                      fkey_col_to_pkey_table={},time_col=None)

        tables["relazioni_societarie"] = Table(df=pd.DataFrame(relazioni_societarie), pkey_col="id",
                                      fkey_col_to_pkey_table={"company_id": "azienda","group_id": "gruppo"},
                                               time_col="date_from"
                                                    )
        return Database(tables)

dataset=Group()

db =dataset.make_db()


print("Tabelle aziende \n",db.table_dict["azienda"].df ,"\n")
print("Tabelle gruppi \n",db.table_dict["gruppo"].df ,"\n")
print("Tabelle relazioni societarie \n",db.table_dict["relazioni_societarie"].df ,"\n")


register_dataset("rel-custom_group",Group)


class GroupMembershipPredictionTask(EntityTask):
    """Task: Predire se un'azienda entrerà o uscirà da un gruppo nei prossimi 6 mesi."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "company_id"
    entity_table = "azienda"
    time_col = "date_from"
    target_col = "variation"
    timedelta = pd.Timedelta(days=60)  # 6 mesi
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40


    def make_table(self, db: Database, timestamps: "pd.Series[pd.timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        gruppo = db.table_dict["gruppo"].df
        company = db.table_dict["azienda"].df
        membership = db.table_dict["relazioni_societarie"].df


        df = duckdb.sql(f"""
  SELECT distinct company_id,date_from,
         CASE WHEN NESSUNA_VARIAZIONE =1 THEN 0 ELSE 1 END variation         
  FROM(
  SELECT 
                company_id,
                group_id,
                DATE_FROM,
                DATE_TO,
                count (group_id) OVER (PARTITION BY company_id) AS NUMERO_PARTECIPAZIONE_GRUPPI,
                CASE WHEN (DATE_TO) IS NULL THEN 1 ELSE 0 END AS NESSUNA_VARIAZIONE 
                
                
            FROM timestamp_df t
             LEFT JOIN membership rs ON
                rs.date_from > t.timestamp
                and rs.date_from < t.timestamp + INTERVAL '{self.timedelta}'
                
                order by company_id
                )
                where company_id IS NOT NULL
order by company_id
""").df()


        return Table(
            df=df,
            time_col=self.time_col,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None
        )


#richiamo il mio dataset custom

##reg_group_dataset=get_dataset("rel-custom_group")


task= GroupMembershipPredictionTask(dataset)




train_table=task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")
out_channels = 1
loss_fn = BCEWithLogitsLoss()
tune_metric = "mae"
higher_is_better = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


col_to_stype_dict = get_stype_proposal(db)


print(train_table,"\n")
print (col_to_stype_dict,"\n")


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))

text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(), batch_size=256
)

data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,  # speficied column types
    text_embedder_cfg=text_embedder_cfg  # our chosen text encoder

      # store materialized graph for convenience
)

print(data)


# caricamento


loader_dict = {}

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table)
]:
    table_input = get_node_train_table_input(
        table=table,
        task=task
    )
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[
            128 for i in range(2)
        ],  # we sample subgraphs of depth 2, 128 neighbors per node.
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=512,
        temporal_strategy="uniform",
        shuffle=split == "train",
        num_workers=0,
        persistent_workers=False
    )


#######


### model


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict
        )

        return self.head(x_dict[dst_table])


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=128,
    out_channels=1,
    aggr="sum",
    norm="batch_norm"
).to(device)


# if you try out different RelBench tasks you will need to change these
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 10



####




#stantard train/test loops #

def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()



##



# training

state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, val_table)
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

#








