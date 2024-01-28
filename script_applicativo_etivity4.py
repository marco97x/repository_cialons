from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer,String,Float,MetaData,ForeignKey,DateTime,Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
from datetime import datetime


engine =create_engine('mysql+mysqlconnector://root:mysqlpwd2023!@127.0.0.1/db')
connection= engine.connect()
Base = declarative_base() 
Session = sessionmaker(bind = engine)
session = Session()


class Impiegato(Base):
    __tablename__='impiegato'
    matricola=Column('matricola',Integer, primary_key = True)
    nome=Column('nome',String(20))
    cognome=Column('cognome', String(20))
    stipendio=Column('stipendio',Integer)
    ruolo= Column('ruolo', String(50))
    progetto_settore_appartenenza= Column('progetto_settore_appartenenza',String(20))
    linea_business_appartenenza= Column('linea_business_appartenenza', String(20))
    


class Partner(Base):
    __tablename__='partner'
    id_partner=Column('Id_partner',Integer,primary_key=True)
    nome_partner=Column('nome_partner',String(20)) 
    cognome_partner=Column('cognome_partner',String(20))
    quote_azienda=Column('quote_azienda',Float)
    business_line_riferimento=Column('business_line_riferimento',String(20))
    
class line_business(Base):
    __tablename__='line_business'
    codice_linea_business=Column('codice_linea_business',Integer,primary_key=True)
    nome_linea_business=Column('nome_linea_business',String(20))
    tipo_tecnologia=Column('tipo_tecnologia',String(20)) 
    id_partner=Column('id_partner',Integer,ForeignKey('partner.Id_partner'))
    matricola_manager=Column('matricola_manager',Integer)
    nome_partner_riferimento=Column('nome_partner_riferimento',String(20))
    cognome_partner_riferimento=Column('cognome_partner_riferimento',String(20))
    nome_manager_riferimento=Column('nome_manager_riferimento',String(20))
    cognome_manager_riferimento=Column('cognome_manager_riferimento',String(20))
    partner_riferimento = relationship("Partner",back_populates="lob") 

class cliente(Base):
    __tablename__='cliente'
    Id_cliente= Column('id_cliente',Integer,primary_key=True)
    nome_cliente=Column('nome_cliente',String(20))
    azienda_cliente=Column('azienda_cliente',String(40))
    ruolo =Column('ruolo',String(40))
    progetto=Column('progetto',String(40))

class progetto(Base):
    __tablename__='progetto'
    id_progetto= Column('id_progetto',Integer,primary_key=True) 
    nome_progetto=Column('nome_progetto',String(20))
    id_referente_cliente=Column('id_referente_cliente',Integer)
    nome_cliente=Column('nome_cliente',String(20))
    nome_manager_riferimento=Column('nome_manager_riferimento',String(20)) 
    cognome_manager_riferimento=Column('cognome_manager_riferimento',String(20))
    budget_progetto=Column('budget_progetto',Integer)
    numero_persone_progetto=Column('numero_persone_progetto',Integer)
    data_inizio=Column('data_inizio',DateTime)
    data_fine= Column('data_fine',DateTime)
    codice_business_line_progetto=Column('codice_business_line_progetto',Integer,ForeignKey('line_business.codice_linea_business'))
    business_line=relationship("line_business",back_populates="progetto")

    

class collaboratori_esterni(Base):
    __tablename__='collaboratori_esterni'
    id_collaboratore=Column('id_collaboratore',Integer,primary_key=True)
    nome=Column('nome',String(40))
    cognome=Column('cognome',String(40))
    servizio_offerto=Column('servizio_offerto',String(40))
    azienda_collaboratore=Column('azenda_collaboratore',String(40))
    id_progetto_collaborazione=Column('id_progetto_collaborazione',String(20))


class aree_amministrative(Base):
    __tablename__='aree_amministrative'
    cod_area=Column('cod_area',Integer,primary_key=True)
    settore_amministrativo=Column('settore_amministrativo',String(20))
    sede_amministrazione=Column('sede_amministrazione',String(20))
    

class soci(Base):
    __tablename__='soci'
    cod_azienda_socio=Column('cod_azienda_socio',Integer,primary_key=True)
    azienda_socio=Column('azienda_socio',String(20))
    partecipazione_gruppo=Column('partecipazione_gruppo',String(20))
    capitale_sociale=Column('capitale_sociale',Integer)


class sedi(Base):
    __tablename__='sedi'
    id_sede=Column('id_sede',Integer,primary_key=True)
    nome_sede=Column('nome_sede',String(20))
    città=Column('città',String(20))
    indirizzo=Column('indirizzo',String(20))
    numero_dipendenti_sede=Column('numero_dipendenti_sede',Integer)


class azienda(Base):
    __tablename__='azienda'
    id_partner_sede=Column('id_partner_sede',Integer,primary_key=True)
    cod_area_amminisitrativa=Column('cod_area_amminisitrativa',Integer,ForeignKey('aree_amministrative.cod_area'),primary_key=True)
    id_sede= Column('id_sede',Integer,ForeignKey('sedi.id_sede'),primary_key=True)
    nome_sede=Column('nome_sede',String(20))
    nome_area_amministrativa=Column('nome_area_amministrativa',String(20))
    nome_partner_riferimento_sede=Column('nome_partner_riferimento',String(20))
    cognome_partner_riferimento_sede=Column('cognome_partner_riferimento_sede',String(20))
    


Base.metadata.create_all(engine)

line_business.progetto=relationship("progetto",order_by=progetto.id_progetto,
back_populates="business_line")
Partner.lob= relationship("line_business",
                        order_by=line_business.codice_linea_business,
                        back_populates="partner_riferimento")


impiegati=[
Impiegato(matricola='000001',nome='marco',cognome='cialone',stipendio=30000,
          ruolo='consulente',progetto_settore_appartenenza='progetto1',linea_business_appartenenza='line_business1'),
Impiegato(matricola='000002',nome='paolo',cognome='bianchi',stipendio=40000,
          ruolo='manager',progetto_settore_appartenenza='progetto1',linea_business_appartenenza='line_business1'),
Impiegato(matricola='000003',nome='giuseppe',cognome='rossi',stipendio=30000,
          ruolo='impiegato_amministrazione',progetto_settore_appartenenza='settore1',linea_business_appartenenza='line_business1'),
Impiegato(matricola='000004',nome='mario',cognome='verdi',stipendio=30000,
          ruolo='impiegato_amministrazione',progetto_settore_appartenenza='settore2',linea_business_appartenenza='line_business1'),
Impiegato(matricola='000005',nome='luciano',cognome='neri',stipendio=45000,
          ruolo='manager',progetto_settore_appartenenza='progetto2',linea_business_appartenenza='line_business2'),
Impiegato(matricola='000006',nome='francesco',cognome='viola',stipendio=45000,
          ruolo='manager',progetto_settore_appartenenza='progetto3',linea_business_appartenenza='line_business3'),
Impiegato(matricola='000007',nome='luca',cognome='gallo',stipendio=30000,
          ruolo='consulente',progetto_settore_appartenenza='progetto2',linea_business_appartenenza='line_business2'),
Impiegato(matricola='000008',nome='martina',cognome='conti',stipendio=30000,
          ruolo='consulente',progetto_settore_appartenenza='progetto3',linea_business_appartenenza='line_business3')
]



partner=[
Partner(id_partner=1,nome_partner='giovanni',cognome_partner='verdi',quote_azienda=0.3,business_line_riferimento='line_business1'),
Partner(id_partner=2,nome_partner='lorenzo',cognome_partner='marrone',quote_azienda=0.2,business_line_riferimento='line_business2'),
Partner(id_partner=3,nome_partner='matteo',cognome_partner='rossi',quote_azienda=0.5,business_line_riferimento='line_business3')
]

lob= [line_business(codice_linea_business=1,nome_linea_business='line_business1',
                    tipo_tecnologia='servizio1',id_partner=1,matricola_manager=2,
                    nome_partner_riferimento='giovanni',cognome_partner_riferimento='verdi',
                    nome_manager_riferimento='paolo',cognome_manager_riferimento='bianchi'),
line_business(codice_linea_business=2,nome_linea_business='line_business2',
              tipo_tecnologia='servizio2',id_partner=2,matricola_manager=5,nome_partner_riferimento='lorenzo',
              cognome_partner_riferimento='marrone',nome_manager_riferimento='luciano',
              cognome_manager_riferimento='neri'),
line_business(codice_linea_business=3,nome_linea_business='line_business3',
              tipo_tecnologia='servizio3',id_partner=3,matricola_manager=6,nome_partner_riferimento='matteo',
              cognome_partner_riferimento='rossi',nome_manager_riferimento='francesco',cognome_manager_riferimento='viola')]


clienti=[
cliente(Id_cliente='000001',nome_cliente='cliente1',azienda_cliente='azienda_cliente1',
        ruolo='analista',progetto='progetto1'),
cliente(Id_cliente='000002',nome_cliente='cliente2',azienda_cliente='azienda_cliente2',
        ruolo='analista',progetto='progetto2'),
cliente(Id_cliente='000003',nome_cliente='cliente3',azienda_cliente='azienda_cliente3',
        ruolo='analista',progetto='progetto3')
]

progetti=[
    progetto(nome_progetto='progetto1',id_referente_cliente=1,nome_cliente='cliente1', 
        budget_progetto=100000,numero_persone_progetto=3,nome_manager_riferimento='Paolo',
        cognome_manager_riferimento='Bianchi',
        data_inizio=datetime.strptime('01/01/2023', '%m/%d/%Y'),
        data_fine=datetime.strptime('12/31/2023', '%m/%d/%Y'),
        codice_business_line_progetto=1),
progetto(nome_progetto='progetto2',id_referente_cliente=2,nome_cliente='cliente2',
        budget_progetto=200000,numero_persone_progetto=1,nome_manager_riferimento='Luciano',
        cognome_manager_riferimento='Neri',
        data_inizio=datetime.strptime('01/01/2023', '%m/%d/%Y'),
        data_fine=datetime.strptime('12/31/2023', '%m/%d/%Y'),
        codice_business_line_progetto=2
        ),
progetto(nome_progetto='progetto3',id_referente_cliente=3,nome_cliente='cliente3',
         budget_progetto=300000,numero_persone_progetto=3,nome_manager_riferimento='Francesco',
         cognome_manager_riferimento='Viola',
         data_inizio=datetime.strptime('01/01/2023', '%m/%d/%Y'),
         data_fine=datetime.strptime('12/31/2023', '%m/%d/%Y'),
         codice_business_line_progetto=3
         )
]


collaboratori=[collaboratori_esterni(
nome='nome_collaboratore1',cognome='cognome_collaboratore1',servizio_offerto='servizio_offerto1',
azienda_collaboratore='azienda_collaboratore1',id_progetto_collaborazione=1
),
collaboratori_esterni(
nome='nome_collaboratore2',cognome='cognome_collaboratore2',servizio_offerto='servizio_offerto2',
azienda_collaboratore='azienda_collaboratore2',id_progetto_collaborazione=2
),
collaboratori_esterni(
nome='nome_collaboratore3',cognome='cognome_collaboratore3',servizio_offerto='servizio_offerto3',
azienda_collaboratore='azienda_collaboratore3',id_progetto_collaborazione=3
)]

aree_amm=[
aree_amministrative(cod_area=1,settore_amministrativo='legale',sede_amministrazione='sede1'),
aree_amministrative(cod_area=2,settore_amministrativo='contabile',sede_amministrazione='sede1'),
aree_amministrative(cod_area=3,settore_amministrativo='risorse_umane',sede_amministrazione='sede2')
]



soci_azienda=[ 
soci(cod_azienda_socio='1',azienda_socio='azienda_socio1',partecipazione_gruppo=0.2,capitale_sociale=2000000),
soci(cod_azienda_socio='2',azienda_socio='azienda_socio2',partecipazione_gruppo=0.3,capitale_sociale=3000000),
soci(cod_azienda_socio='3',azienda_socio='azienda_socio3',partecipazione_gruppo=0.4,capitale_sociale=4000000),
soci(cod_azienda_socio='4',azienda_socio='azienda_socio3',partecipazione_gruppo=0.1,capitale_sociale=1000000)
]


sedi_azienda=[
sedi(
id_sede=1,nome_sede='sede1',città='Roma',indirizzo='via x1',numero_dipendenti_sede=200
),
sedi(
id_sede=2,nome_sede='sede2',città='Roma',indirizzo='via x2',numero_dipendenti_sede=100
),
sedi(
id_sede=3,nome_sede='sede3',città='Roma',indirizzo='via x3',numero_dipendenti_sede=300
)
]

amministrazione_azienda=[
    azienda(id_partner_sede=1,cod_area_amminisitrativa=1,id_sede=1,nome_sede='sedi1',
            nome_area_amministrativa='legale',nome_partner_riferimento_sede='giovanni',
            cognome_partner_riferimento_sede='verdi'),
azienda(id_partner_sede=2,cod_area_amminisitrativa=2,id_sede=1,nome_sede='sedi1',
        nome_area_amministrativa='contabile',nome_partner_riferimento_sede='giovanni',
        cognome_partner_riferimento_sede='verdi'),
azienda(id_partner_sede=3,cod_area_amminisitrativa=3,id_sede=2,nome_sede='sedi2',
        nome_area_amministrativa='risorse_umane',nome_partner_riferimento_sede='lorenzo',
        cognome_partner_riferimento_sede='marrone')  
    
]

#aggiunta di un nuovo partner che è il riferimento di una nuova line_business 
# con due nuovi progetti associati ad essa

partner_lob_progetto=Partner(
    nome_partner="Tiziano" ,
    cognome_partner="Rossi",
    quote_azienda=0.5,
    business_line_riferimento="line_business5",
    lob=[line_business(
        codice_linea_business=5,
        nome_linea_business="line_business5",
        tipo_tecnologia="servizio5" ,
        matricola_manager="000012",
        nome_partner_riferimento="Tiziano" ,
        cognome_partner_riferimento="Rossi",
        nome_manager_riferimento="Filippo",
        cognome_manager_riferimento="Rossi",
        progetto=[progetto(
        id_progetto=5,
        nome_progetto="progetto5",
        id_referente_cliente=4, 
        nome_cliente="cliente4",
        budget_progetto=200000,
        numero_persone_progetto=3,
        nome_manager_riferimento="Filippo",
        cognome_manager_riferimento="Rossi",
        data_inizio="2024-01-01",
        data_fine="2024-12-31"),
        progetto(
        nome_progetto="progetto6",
        id_referente_cliente=4, 
        nome_cliente="cliente4",
        budget_progetto=200000,
        numero_persone_progetto=3,
        nome_manager_riferimento="Filippo",
        cognome_manager_riferimento="Rossi",
        data_inizio="2024-01-01",
        data_fine="2024-12-31"
        )]

        )]
    )

elenco=[impiegati,partner,lob,clienti,progetti,collaboratori,aree_amm,soci_azienda,sedi_azienda,
        amministrazione_azienda]

for item in elenco:
    session.add_all(item)
    session.commit()
    
session.add(partner_lob_progetto)
session.commit()

# cambio quota partner 
session.query(Partner).filter(Partner.id_partner==3).update({Partner.quote_azienda:0.25})
session.query(Partner).filter(Partner.id_partner==4).update({Partner.quote_azienda:0.25})


progetti_terminati=session.query(progetto).filter(progetto.data_fine.like('2023%')).all()

for i in progetti_terminati:
    print (i.nome_progetto,i.nome_cliente,i.data_inizio,i.data_fine)

totale_progetti=session.query(progetto)

for i in totale_progetti:
    print ("Elenco completo dei Progetti pre eliminazione progetti scaduti:/n",i.nome_progetto,i.nome_cliente,i.data_inizio,i.data_fine,"\n")

for i in progetti_terminati:
    session.delete(i)

session.commit()

for i in totale_progetti:
    print ("Elenco completo dei Progetti post eliminazione:/n",i.nome_progetto,i.nome_cliente,i.data_inizio,i.data_fine,"\n")


#session.add_all([partner_new,impiegato_new1,impiegato_new2,impiegato_manager_new1,cliente_new,line_business_new,progetto_new])















