Search.setIndex({docnames:["README","advanced","advanced/saving","apidocs/padl","apidocs/padl.dumptools","apidocs/padl.dumptools.ast_utils","apidocs/padl.dumptools.inspector","apidocs/padl.dumptools.packagefinder","apidocs/padl.dumptools.serialize","apidocs/padl.dumptools.sourceget","apidocs/padl.dumptools.symfinder","apidocs/padl.dumptools.var2mod","apidocs/padl.exceptions","apidocs/padl.print_utils","apidocs/padl.transforms","apidocs/padl.util_transforms","apidocs/padl.utils","apidocs/padl.version","apidocs/padl.wrap","gettingstarted","index","modules","usage","usage/apply","usage/combining_transforms","usage/creating_transforms","usage/extras","usage/print_slice","usage/pytorch","usage/saving","usage/stages","usage/transform"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["README.md","advanced.md","advanced/saving.md","apidocs/padl.md","apidocs/padl.dumptools.md","apidocs/padl.dumptools.ast_utils.md","apidocs/padl.dumptools.inspector.md","apidocs/padl.dumptools.packagefinder.md","apidocs/padl.dumptools.serialize.md","apidocs/padl.dumptools.sourceget.md","apidocs/padl.dumptools.symfinder.md","apidocs/padl.dumptools.var2mod.md","apidocs/padl.exceptions.md","apidocs/padl.print_utils.md","apidocs/padl.transforms.md","apidocs/padl.util_transforms.md","apidocs/padl.utils.md","apidocs/padl.version.md","apidocs/padl.wrap.md","gettingstarted.md","index.md","modules.md","usage.md","usage/apply.md","usage/combining_transforms.md","usage/creating_transforms.md","usage/extras.md","usage/print_slice.md","usage/pytorch.md","usage/saving.md","usage/stages.md","usage/transform.md"],objects:{"":[[3,0,0,"-","padl"]],"padl.dumptools":[[5,0,0,"-","ast_utils"],[6,0,0,"-","inspector"],[7,0,0,"-","packagefinder"],[8,0,0,"-","serialize"],[9,0,0,"-","sourceget"],[10,0,0,"-","symfinder"],[11,0,0,"-","var2mod"]],"padl.dumptools.ast_utils":[[5,1,1,"","Position"],[5,2,1,"","get_position"],[5,2,1,"","get_source_segment"]],"padl.dumptools.inspector":[[6,1,1,"","CallInfo"],[6,2,1,"","caller_frame"],[6,2,1,"","caller_module"],[6,2,1,"","get_segment_from_frame"],[6,2,1,"","get_statement"],[6,2,1,"","get_surrounding_block"],[6,2,1,"","non_init_caller_frameinfo"],[6,2,1,"","outer_caller_frameinfo"],[6,2,1,"","trace_this"]],"padl.dumptools.inspector.CallInfo":[[6,3,1,"","module"]],"padl.dumptools.packagefinder":[[7,2,1,"","dump_packages_versions"],[7,2,1,"","get_packages"],[7,2,1,"","get_version"]],"padl.dumptools.serialize":[[8,1,1,"","Serializer"],[8,2,1,"","json_serializer"],[8,2,1,"","load_json"],[8,2,1,"","save_json"],[8,2,1,"","value"]],"padl.dumptools.serialize.Serializer":[[8,4,1,"","save"],[8,4,1,"","save_all"],[8,3,1,"","varname"]],"padl.dumptools.sourceget":[[9,1,1,"","ReplaceString"],[9,1,1,"","ReplaceStrings"],[9,2,1,"","cut"],[9,2,1,"","get_module_source"],[9,2,1,"","get_source"],[9,2,1,"","original"],[9,2,1,"","put_into_cache"]],"padl.dumptools.sourceget.ReplaceString":[[9,4,1,"","cut"]],"padl.dumptools.sourceget.ReplaceStrings":[[9,4,1,"","cut"]],"padl.dumptools.symfinder":[[10,5,1,"","NameNotFound"],[10,1,1,"","Scope"],[10,1,1,"","ScopedName"],[10,2,1,"","find"],[10,2,1,"","find_in_ipython"],[10,2,1,"","find_in_module"],[10,2,1,"","find_in_scope"],[10,2,1,"","find_in_source"],[10,2,1,"","replace_star_imports"],[10,2,1,"","split_call"]],"padl.dumptools.symfinder.Scope":[[10,4,1,"","empty"],[10,4,1,"","from_level"],[10,4,1,"","from_source"],[10,4,1,"","global_"],[10,4,1,"","is_global"],[10,3,1,"","module_name"],[10,4,1,"","toplevel"],[10,4,1,"","unscoped"],[10,4,1,"","up"]],"padl.dumptools.var2mod":[[11,1,1,"","CodeGraph"],[11,1,1,"","CodeNode"],[11,1,1,"","Finder"],[11,1,1,"","Vars"],[11,2,1,"","find_codenode"],[11,2,1,"","find_globals"],[11,2,1,"","increment_same_name_var"],[11,2,1,"","rename"]],"padl.dumptools.var2mod.CodeGraph":[[11,4,1,"","build"],[11,4,1,"","dumps"],[11,4,1,"","print"]],"padl.dumptools.var2mod.CodeNode":[[11,4,1,"","from_source"]],"padl.dumptools.var2mod.Finder":[[11,4,1,"","find"],[11,4,1,"","generic_visit"],[11,4,1,"","get_source_segments"]],"padl.dumptools.var2mod.Vars":[[11,6,1,"","globals"],[11,6,1,"","locals"]],"padl.exceptions":[[12,5,1,"","WrongDeviceError"]],"padl.print_utils":[[13,2,1,"","combine_multi_line_strings"],[13,2,1,"","create_arrow"],[13,2,1,"","create_reverse_arrow"],[13,2,1,"","format_argument"],[13,2,1,"","make_bold"],[13,2,1,"","make_green"],[13,2,1,"","visible_len"]],"padl.transforms":[[14,1,1,"","AtomicTransform"],[14,1,1,"","Batchify"],[14,1,1,"","BuiltinTransform"],[14,1,1,"","ClassTransform"],[14,1,1,"","Compose"],[14,1,1,"","FunctionTransform"],[14,1,1,"","Identity"],[14,1,1,"","Map"],[14,1,1,"","Parallel"],[14,1,1,"","Pipeline"],[14,1,1,"","Rollout"],[14,1,1,"","TorchModuleTransform"],[14,1,1,"","Transform"],[14,1,1,"","Unbatchify"],[14,2,1,"","fulldump"],[14,2,1,"","group"],[14,2,1,"","importdump"],[14,2,1,"","load"],[14,2,1,"","save"]],"padl.transforms.ClassTransform":[[14,3,1,"","source"]],"padl.transforms.FunctionTransform":[[14,3,1,"","source"]],"padl.transforms.Pipeline":[[14,4,1,"","grouped"],[14,4,1,"","pd_forward_device_check"],[14,4,1,"","pd_to"]],"padl.transforms.TorchModuleTransform":[[14,4,1,"","post_load"],[14,4,1,"","pre_save"]],"padl.transforms.Transform":[[14,4,1,"","eval_apply"],[14,4,1,"","infer_apply"],[14,4,1,"","pd_call_transform"],[14,3,1,"","pd_device"],[14,3,1,"","pd_forward"],[14,4,1,"","pd_forward_device_check"],[14,4,1,"","pd_get_loader"],[14,3,1,"","pd_layers"],[14,3,1,"","pd_name"],[14,4,1,"","pd_parameters"],[14,4,1,"","pd_post_load"],[14,3,1,"","pd_postprocess"],[14,4,1,"","pd_pre_save"],[14,3,1,"","pd_preprocess"],[14,4,1,"","pd_save"],[14,4,1,"","pd_set_mode"],[14,4,1,"","pd_to"],[14,4,1,"","pd_varname"],[14,4,1,"","pd_zip_save"],[14,4,1,"","train_apply"]],"padl.util_transforms":[[15,1,1,"","IfEval"],[15,1,1,"","IfInMode"],[15,1,1,"","IfInfer"],[15,1,1,"","IfTrain"],[15,1,1,"","Try"]],"padl.utils":[[16,6,1,"","same"]],"padl.wrap":[[18,1,1,"","PatchedModule"],[18,2,1,"","transform"]],padl:[[3,1,1,"","Batchify"],[3,1,1,"","Identity"],[3,1,1,"","IfEval"],[3,1,1,"","IfInMode"],[3,1,1,"","IfInfer"],[3,1,1,"","IfTrain"],[3,1,1,"","Unbatchify"],[4,0,0,"-","dumptools"],[12,0,0,"-","exceptions"],[3,2,1,"","fulldump"],[3,2,1,"","group"],[3,2,1,"","importdump"],[3,2,1,"","load"],[13,0,0,"-","print_utils"],[3,2,1,"","save"],[3,2,1,"","transform"],[14,0,0,"-","transforms"],[3,7,1,"","unbatch"],[15,0,0,"-","util_transforms"],[16,0,0,"-","utils"],[3,2,1,"","value"],[17,0,0,"-","version"],[18,0,0,"-","wrap"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","function","Python function"],"3":["py","property","Python property"],"4":["py","method","Python method"],"5":["py","exception","Python exception"],"6":["py","attribute","Python attribute"],"7":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:function","3":"py:property","4":"py:method","5":"py:exception","6":"py:attribute","7":"py:data"},terms:{"0":[0,3,6,9,10,11,14,19,26,27],"02_nlp_exampl":19,"0x":11,"1":[0,2,3,9,10,11,14,15,19,24,26,27,28],"10":[0,3,15,19,25],"100":[2,11,24,25,26,27],"1000":25,"11":[3,6,15],"123":6,"13":9,"2":[0,3,9,10,11,14,19,24,26,27],"20":0,"200":26,"244":24,"3":[0,3,5,6,10,11,14,19,24,26],"300":26,"4":[11,26],"5":9,"512":19,"64":19,"8":[0,5,19],"9":[0,9,19],"99":[3,15],"abstract":[0,14,19,20,31],"case":[2,3,9,10,14,24],"catch":[15,22],"class":[2,3,5,6,8,9,10,11,14,15,18,19,25,30],"const":2,"default":[2,3,6,7,8,10,14,15,24],"do":[2,3,12,14,22,23,25],"final":[15,19,26],"function":[0,2,3,6,8,10,11,14,18,19,24,25,31],"import":[0,2,3,7,10,11,14,15,18,19,20,24,25,26,29,30],"int":[5,6,9,10,11,13,14],"new":[9,10],"return":[0,2,6,7,8,9,10,11,14,19,23,24,25,26,27,30],"static":14,"super":19,"switch":[2,3,14],"true":[3,7,9,10,11,13,14,18,19,24,27],"try":[3,6,9,14,15,18,22,26],"var":11,A:[3,6,9,10,11,14,19,31],As:[2,11,30],By:24,For:[2,19,24,25,26],If:[0,2,3,8,9,10,11,13,14,15,19],In:[3,9,10,14,19,24,25],It:[2,6,10,14,23,25,28],No:15,One:[2,24],The:[2,3,6,8,9,10,11,13,14,15,19,22,24,30,31],These:[2,14,19],To:[2,3,9,14,19,23,24,30],With:31,_:[2,19],____________:13,__call__:[2,19,25],__file__:2,__getitem__:19,__init__:[2,6,19,25],__main__:10,_ast:[10,11],_pd_main:2,_thingfind:10,_trace_thi:6,a_transform:2,about:[6,14,28,31],abov:[19,24],absolut:2,accept:14,access:[10,19,27,30],ad:[3,10,14,23,31],adam:[19,28],adapt:28,add:[2,3,9,14,25,26,27],add_n:2,addconst:2,addfirst:26,addit:9,advanc:20,after:[2,3,14,28],again:2,ai:0,airplan:23,alia:11,all:[0,2,6,8,11,12,14,18,19,22,23,24,28,29],allow:[0,9,10,19,24,26],along:28,also:[2,3,14,19,23,25,26],alwai:14,an:[0,2,6,9,10,11,13,14,19,23,26,28,31],analog:26,ani:[3,8,13,14,15,19],anoth:13,anyth:[2,20],apach:0,api:20,append:[2,8,19],appli:[2,3,13,14,15,20,22,24,30],applic:26,ar:[2,3,8,10,11,14,19,22,24,25,28,29,30],arbitrari:[0,26],arg:[2,6,14,19,23,27],argument:[2,6,10,13,14,23],aris:24,arrai:[2,11,26],arrow:13,ascii:13,assign:[10,14],ast:[5,7,10,11],ast_nod:11,ast_util:[20,21],atom:14,atomictransform:14,attribut:[6,9,16],augment:[24,26,28,30],automat:[2,3,14,23],avail:[6,23],awar:2,axxxxxxxxxxxxxxxxxxxx:9,b:[3,9,11,14,15,19],back:15,backward:[19,28],bar:[11,14,24],base:[0,8,14],batch:[0,3,14,19,22,23,24,28,30,31],batch_first:19,batch_siz:[19,23,28],batchifi:[3,10,14,22],baz:24,becom:[6,10,19,24,25],been:14,befor:[2,6,14,23],being:[3,14,26],below:11,berlin:0,between:[11,24],big:2,bla:7,blip:7,blob:0,block:[6,19],blop:7,blu:7,bodi:[2,6,7,10,11],boilerpl:31,bold:13,bool:[3,6,9,10,11,14],both:[2,9,28],branch:19,build:[0,2,10,11,31],build_my_transform:2,builder:0,built:[19,28],builtin:14,builtintransform:14,bup:7,c:[3,14],cach:9,call:[3,6,8,10,11,14,15,25,30],call_info:14,call_sourc:10,callabl:[3,6,8,14,18,19,25],caller:6,caller_fram:6,caller_modul:6,callinfo:[6,14],calling_scop:10,can:[0,2,3,6,9,10,14,15,19,23,24,25,26,27,28,29,30,31],canb:23,cannot:2,capabl:28,captur:16,carri:15,cat:[19,23],catch_transform:15,caus:2,caveat:2,cell:9,central:31,chang:[2,6,15],charact:13,check:[14,19],checkpoint:29,chief:19,child:[12,14],child_transform:12,children:14,citizen:19,classifi:[24,30],classmethod:[8,10,11],classtransform:14,claus:15,clean:[2,19],close:26,co:19,code:[0,2,3,6,7,8,9,10,11,14,28,31],code_of_conduct:0,codegraph:[8,11],codenod:11,col:[6,9,11],col_offset:5,collect:[9,14,22],column:6,com:[0,19],combin:[13,14,19,20,22,25,31],combine_multi_line_str:13,come:[18,25],comment:6,common:24,complet:[6,15],complex:[2,14,27,31],compon:[0,19,24],compos:[14,19,27,30],composit:[19,24],comprehens:[0,19],compress:[3,14],comput:19,concat_low:26,concis:[0,19,22],condit:[15,22],condition:24,conduct:0,conflict:2,consid:2,consist:2,constant:[2,9],construct:14,contain:[2,6,8,9,10,12,13,14,28,29],content:11,context:[6,14],continu:[24,30],contrast:14,contribut:20,conveni:28,convert:[0,10,25,28,30],coor:13,coordin:13,correct:14,correspond:[5,10,11,30],cosin:19,could:[2,10,24],count:10,counter:[10,11],cpu:[3,14],creat:[2,6,8,10,11,13,14,20,22,23,24,28,29,31],create_arrow:13,create_reverse_arrow:13,crop:24,cuda:[14,28],current:[0,2,10,19],custom:[14,31],cut:9,data:[2,14,20,22,30,31],dataload:[14,23,31],datapoint:[14,24],dataset:2,debug:[11,31],decor:[3,14,18,19,25],deep:[0,2,19,20,24,30,31],def:[0,2,6,10,11,14,19,25,26,27,30],def_sourc:10,defin:[8,10,11,22,29,30],definit:[2,19],depend:[2,11,29],detail:[6,14,19,24],determin:[3,6,9,11,14,18],develop:0,devic:[3,12,14,23],di:31,dict:[9,19],dictionari:[2,14,19],differ:[2,9,23,30],dim:[3,14],dimens:[3,14,23,31],directli:14,directori:2,disabl:[3,13,14,23],document:[6,14],doe:[14,23],doesn:[3,15],dog:[19,23],don:[2,3,6,13,14,18],done:[14,23],down:2,drop:[6,10],drop_n:[6,10],dump:[2,3,7,8,11,14],dump_packages_vers:7,dumptool:[14,20,21],dure:[3,14,24,26],dynam:19,e:[3,10,14,19],each:[2,14,19,23,30],easili:31,ecosystem:0,edg:11,effect:2,either:[2,9],eleg:31,element:[14,19,24,30],els:[3,11,15],else_:[3,15],else_transform:15,emb:19,embed:19,emit:19,empti:[6,10],enabl:[2,3,14,31],end:[3,11,13,14],end_col_offset:5,end_lineno:5,enter:6,entir:25,entiti:10,entri:2,enumer:2,eos_valu:19,equival:26,error:15,escap:13,eval:[3,14,15,22,23,26],eval_appli:[3,14,15,19,23,26],evalu:30,even:24,event:6,everi:13,everyth:[11,22,24,25],exactli:24,exampl:[0,2,3,6,7,9,10,11,13,14,15,18,19,23,26,30],except:[10,15,20,21,22],execut:[2,6,24],exist:[3,11,14],expect:[2,3,9,14,23],explicit:[9,11],explicitli:14,extens:[3,14],extra:[15,20,22,28],extract:6,f1:14,f2:14,f:[2,9,10,11,14],factori:16,fail:15,fall:15,fals:[3,6,10,11,13,14,18],few:2,field:11,file:[2,3,9,14,26,29],file_suffix:8,fileexistserror:[3,14],filenam:[2,9,14,26],filter_builtin:11,finally_transform:15,find:[0,7,10,11,19],find_codenod:11,find_glob:11,find_in_ipython:10,find_in_modul:10,find_in_scop:10,find_in_sourc:10,finder:11,finish_right:13,first:[3,6,9,13,14,19,24],flat:24,flatten:[3,14,24],flexibl:31,folder:[2,3,14,29],follow:[2,6,10,11,22],foo:[7,11,14,24],force_overwrit:[3,14],form:[10,14,25,31],formal:31,format:[7,13,30,31],format_argu:13,forward:[0,14,19,20,22,23,25],forward_pass:19,found:[6,10,11],four:10,frame:[6,9],frameinfo:6,from:[0,3,6,7,9,10,11,14,15,19,20,25,26,28,29,30],from_:11,from_col:9,from_level:10,from_lin:9,from_sourc:[10,11],front:6,full:[0,2,3,14,19],fulldump:[2,3,14],fulli:[3,14],functiontransform:14,fundament:14,futur:2,g:[14,19],gener:[14,23],generic_visit:11,germani:0,get:[2,5,6,7,9,10,11,13,14,16,20,22,24,26],get_module_sourc:9,get_packag:7,get_posit:5,get_segment_from_fram:6,get_sourc:9,get_source_seg:[5,11],get_stat:6,get_surrounding_block:6,get_vers:7,getitem:[6,27],github:[0,19],give:9,given:[3,6,7,9,10,11,14],global:[2,10,11],global_:10,globals_:11,go:[0,11,19],good:14,gotten:9,gpu:[14,23,30,31],gradient:[14,19,23,31],graph:[11,19],great:0,green:13,group:[3,14,22],gru:19,ha:[2,9,11,14,19,24,28,30],hacki:6,handl:23,happen:[15,30],have:[0,14,28],head:24,hello:26,help:[14,19],helper:[3,8],here:[2,6,19],hidden:25,hierarchi:10,histori:10,hood:23,how:[10,14,23,24,25,27,29,30,31],howev:2,http:[0,19],i:[2,3,10,14],id:9,ident:[3,14,15,22,27],identifi:10,if_:[3,15],ifev:[3,15,22,26],iff:10,ifinf:[3,15,22,26],ifinmod:[3,15],iftrain:[3,15,19,22,24,26],ignore_scop:[3,6,14,18],iim:[3,15],imag:[13,30],implement:[14,19,25],importdump:[2,3,14],includ:[2,9,14,19,28,29],increment:11,increment_same_name_var:11,indent:6,index:[14,19],indic:[10,14],individu:[19,24],infer:[3,14,15,22,23,26,28,30],infer_appli:[3,14,15,19,23,26],infer_pipelin:28,inform:[6,14],initi:[2,14,28],inlin:[6,14,19],innov:0,input:[3,9,13,14,23,24,26],insert:9,insid:15,inspect:[6,9,14,31],inspector:[14,20,21],instanc:[2,19,24,25,28],instanti:19,instead:[2,3,6,14,18],instruct:19,interact:[2,29,31],interest:0,intermedi:19,intern:23,introduct:20,invers:[3,14],io:[0,19],ipynb:19,ipython:[2,9,10,27],is_glob:10,isinst:[9,11,18,19],issu:[0,19],item:[0,3,10,14,16,23,26],iter:[14,23,28],its:[10,11,14],itself:6,jpg:23,json:[2,8],json_seri:8,just:[2,3,14],keep:24,kei:9,keyword:[0,14],kic:19,know:22,kwarg:[6,14,19],l1:25,l2:25,lab:0,lambda:[3,14,15,19,24,25,26],larg:[2,19],larger:[6,19],largest:6,last:[2,9,13,14],layer1:28,layer2:28,layer:[0,14,19],learn:[0,19,20,23,24,25,28,29,30,31],left_shift:19,len:[2,19],length:[13,24],level:[2,6,10,24],lf1:[0,19],licens:20,like:[2,19,23,30],limit:2,lin:25,line:[6,9,10,13],linear:[0,19,25],lineno:[5,6,10,11],link:19,list:[2,3,7,9,10,11,13,14,15,23,26,28,29],liter:[3,6,14,15],live:11,lm:19,load:[2,3,8,14,20,22,23,24,28,30],load_data:2,load_funct:8,load_imag:[24,30],load_json:8,loader:[8,14],loc:9,local:11,locat:[6,9],log10:25,look:[0,2,6,10,19],lookup:[28,30],loop:[10,28],loos:19,loss:[0,19,24,28],lower:26,lower_cas:19,lr:[19,28],macro:24,made:[6,14],mai:[0,19],main:[0,10,12,19],maintain:9,make:[0,13,22,28],make_bold:13,make_green:13,man:19,manual:[0,19],map:[14,19],mark:[3,8,14],mat:19,match:[3,6,12,14,15],math:25,md:0,mean:[2,3,14,24],messi:2,method:[6,14,19,22,26],might:[2,19,26],minim:31,minus100:25,minusx:[25,26,27],mlp:25,mml:25,mode:[3,14,15,23],model:[0,2,14,28,30],model_pass:24,modifi:9,modul:[3,6,7,8,9,10,11,14,18,19,20,22,24,25,27,29,30],module_nam:[6,10],more:[2,3,6,14,24,31],most:[10,24,30],mother:12,mother_transform:12,move:[3,14,24],much:31,multi:[6,13,31],multipl:[2,14,19,23,25,28],multiprocess:14,must:[2,6,14,24],my:2,my_classifi:[23,30],my_classifier_transform:[24,30],my_pipelin:[0,2,28,29],myfil:2,myfunct:6,mykei:9,myload:2,mypipelin:29,mypytorchlay:28,mysav:2,mytransform:2,n:[6,10,11,14],n_final_row:13,n_in:25,n_initial_row:13,n_out:25,n_to_add:2,n_word:19,name:[2,3,6,7,10,11,14,15,30],namenotfound:10,natur:23,need:[2,8,14,20],nest:[10,27,28],never:14,next:[19,23,24,25,27,28,29,30,31],nextmodul:6,nn:[0,14,19,25],node:[5,7,10,11],nodetyp:11,nodevisitor:11,non:9,non_init_caller_frameinfo:6,none:[3,6,8,10,11,14,15,18],normal:[2,9,14,19],not_found:2,note:[2,6],notebook:[0,2,19,31],noth:[3,14,22],now:2,np:[2,11,18,25,26],npy:2,nth:24,num_work:[19,23,28],number:[6,10,11,13],numpi:[2,18,19,25],o:[11,19],object:[6,9,10,11,14,19,26],obtain:19,occur:15,offset:[6,11],often:[19,26],one:[2,3,10,13,14,15,19,24,30],onli:[2,11,19,24,26],open:[2,24,26],open_fil:26,oper:[19,24],optim:[19,28],option:[3,6,8,11,14,15],order:14,ordereddict:14,origin:[6,9],other:[14,15],otherwis:[3,14,15],out:[6,9,10,11,15,18,24,25],outer:6,outer_caller_frameinfo:6,output:[3,14,15,19,23,24,30],over:[13,14,19,24,28],overlap:9,overrid:14,overwrit:[3,14],own:2,packag:[0,2,3,7,14,29],packagefind:[20,21],pad:13,padl:[2,19,21,24,25,26,27,29,30,31],padl_value_0:2,page:22,parallel:[3,14,19],param:14,paramet:[2,3,6,7,8,9,10,11,12,13,14,15,18,19,29],parent:[2,18],parenthes:24,pars:[7,11],part:[9,14,19,22,23,30],parti:2,particular:25,pass:[0,2,3,14,23],patch:18,patchedmodul:18,path:[2,3,8,14,24,30],pathlib:[2,3,8,14],pd_:19,pd_call_transform:14,pd_devic:14,pd_forward:[14,23,30],pd_forward_device_check:14,pd_get_load:14,pd_group:14,pd_layer:[14,28],pd_name:[14,15],pd_np:18,pd_paramet:[14,19,28],pd_post_load:14,pd_postprocess:[14,30],pd_pre_sav:14,pd_preprocess:[14,30],pd_save:[14,19],pd_set_mod:14,pd_to:[14,28],pd_varnam:14,pd_zip_sav:14,perform:[3,14,15],pick:2,piec:[10,11],pil:24,piltotensor:24,pip:[0,19],pipelin:[0,14,20,22,23,25,27,28,29,30],plan:19,pleas:0,pluson:2,plustwo:2,point:19,posit:[5,9,11],position:19,possibl:[2,6,14,19,25,28],post:[0,19,30],post_load:14,postprocess:[3,14,19,20,22,23,24],potenti:[2,6,19],power:24,pre:[0,9,19,30],pre_sav:14,preced:13,predict:[19,24,28],prepar:0,preprocess:[3,14,19,20,22,23,24,28,31],preprocess_imag:24,pretti:27,prevent:[2,3,14],previous:2,primit:19,print:[6,11,13,14,19,20,22,29],print_util:[20,21],process:[0,2,19,23,30,31],progress:14,propag:14,properti:[6,8,10,14],provid:[2,24],pt:2,put:9,put_into_cach:9,py:[2,29],pypi:2,python:[0,2,5,6,10,11,19,24,25,29],pytorch:[0,14,19,20,22,23,24,25,27,29,30],rais:[3,6,14],rand:18,random:18,randomresizedcrop:24,randomrot:24,rang:[2,19],rather:[2,6,14],re:[0,24],read:[2,22,23,26,27,31],read_from_path:26,readabl:30,recent:10,recreat:2,recurs:[3,14,19],refer:[19,22],referenc:19,reflex:19,regardless:15,releas:19,relu:25,rememb:28,remov:[3,6,14,23],renam:[2,11],rename_loc:11,repeat:[2,24],repl:9,replac:[9,10],replace_cach:9,replace_star_import:10,replacestr:9,repres:[10,11,19],reproduc:31,requir:[2,6],reshap:30,resiz:24,resnet18:[24,30],respect:[3,11,14,26],respons:[2,30],rest:6,result:[2,3,6,9,14,24],return_loc:6,return_parti:10,revers:2,right_shift:19,rnn:19,roll:24,rollout:[3,14,19],row:6,rstring:9,s:[3,6,7,8,9,10,14,15,19,22,23,25,28,29,30],same:[3,5,6,9,10,11,14,16,22,24,28],sat:19,save:[1,3,8,14,20,22,23,31],save_al:8,save_funct:8,save_json:8,saver:8,scalar:19,schemat:19,scope:[3,6,10,11,14,18],scoped_nam:11,scopednam:[10,11],scopelist:10,search:[10,11,14],second:[2,3,10,14,24],section:[9,23,24,25,27,28,29,30,31],see:[0,2,3,6,14,19,23],segment:[5,6,10,11],segment_typ:6,self:[2,10,14,19,25],send:[14,23,28,31],sequenc:[13,14,24],seri:14,serial:[3,14,20,21],session:[2,29],set:[2,8,11,14,23],settrac:6,sever:19,sh:19,should:[2,3,6,8,14],similar:2,simpli:[2,14,19],sin:25,singl:[0,14,19,23,24],size:23,slice:[19,20,22,29,31],so:[10,19,24],some:[2,24,28],somemodul:11,someobject:2,someth:[2,26],sometim:2,soon:6,sourc:[5,6,7,9,10,11,14,19],sourceget:[20,21],space:[6,13],special:[24,26,30],specifi:[3,9,10,14,23],split:[2,10,19],split_cal:10,split_str:19,stack:6,stage:[19,20,22,23,24],stand:30,standard:[0,19],star:[2,10],start:[3,10,13,14,20,22],start_left:13,state_dict:28,statement:[2,3,6,10,14,15],step:[14,19,24,28],step_1:19,stop:31,store:[2,3,8,28,29],str:[2,3,6,8,9,10,11,13,14,15],straightforward:19,string:[3,7,8,9,10,11,13,14],sub:[9,11,14,24,27],subclass:10,substr:9,subval:2,successfulli:15,suffix:2,suggest:30,superimpos:13,support:[0,2,19],surround:[6,10],sy:6,symbol:[2,10],symfind:[6,11,20,21],t1:[14,24,27],t2:[14,24,27],t3:[14,24,27],t4:27,t5:27,t:[2,3,6,13,14,15,18,19,24,26],take:[13,24],taken:[9,25],target:[6,24],target_mod:[3,15],tensor:[0,3,14,19,23],term:19,test:19,text:[0,2],textcorpu:2,th:19,than:2,them:[2,3,14,22,24,28],thereof:[3,14],thi:[2,3,6,9,10,11,13,14,15,19,24,25,27,28,29],thing:[2,3,8,9,10,11],third:2,those:[2,11,28,30],thought:19,three:[23,24],through:[11,19,24],thu:[2,24],time:2,to_col:9,to_integ:19,to_lin:9,to_tensor:19,toarrai:2,togeth:[19,27],tointeg:19,token:19,too:[25,27,29],top:[10,13],topic:20,topk:0,toplevel:[3,6,10,14,18],topolog:0,torch:[0,14,19,25,28],torchmoduletransform:[14,19],torchvis:[19,24,30],totensor:[24,30],trace:6,trace_thi:6,tracefunc:6,track:2,train:[3,14,15,22,23,26,28,30],train_appli:[14,19,23,28],train_data:[19,28],train_model:28,train_pipelin:[19,28],training_pipelin:24,transform:[0,3,10,12,15,16,18,20,21,22,27,29,30],transform_1:19,transform_2:19,transform_or_modul:[3,14],transpar:[13,31],tree:[0,10,11,19],tupl:[2,3,6,10,11,14,15,19,24],turn:14,tvt:24,two:2,txt:[2,29],type:[6,11],typic:[14,19],unbatch:[0,3,19,22,24,30],unbatchifi:[3,14],under:[0,23],undo:2,unind:6,union:[3,14,15],uniqu:14,unk:19,unscop:10,unsqueez:[3,14],up:[2,10],upcom:19,updat:23,us:[0,2,3,6,7,8,9,10,11,13,14,15,18,19,20,22,23,24,25,26,27,30,31],usag:[0,20],use_replace_cach:9,user:0,usual:30,util:[6,7,13,14,20,21,24,28],util_transform:[20,21],v:13,val:[2,3,8],valu:[3,8,9,10,13,19,22],valueerror:6,var2mod:[20,21],var_nam:10,variabl:[10,11,14],variou:[9,10],varnam:[8,10],verbos:14,veri:2,version:[0,2,7,9,10,14,20,21,29],via:[2,14,19,27,28,29,30],view:24,visibl:13,visible_len:13,visitor:11,wa:[0,6,14],wai:10,want:[2,26],we:[2,19,30],well:10,were:[2,10],what:[14,19,26,30],whatev:14,when:[2,3,14,15,25,28,29],where:[6,9,11,13],wherea:30,which:[2,3,6,9,10,11,14,19,23,24,29],whole:[14,31],whose:[3,14],wide:6,within:[6,9,10,14,28],without:31,word:[2,19],word_dropout:19,word_index:2,wordindex:2,work:[27,28,31],worker:23,workflow:[19,30,31],worri:31,would:[2,9,10,19,24],wrap:[2,3,14,20,21,25],wrappe:[3,18],wrapper:[3,18,25],write:[0,2,3,14],written:10,wrongdeviceerror:12,x1:14,x2:14,x3:14,x:[0,2,3,9,10,11,13,14,15,19,24,25,26,27,30],xxxx:9,xxxxxaxxxxxxxxxxxxxxxxxxxx:9,xxxxxxxxxxx:9,xxxxxxxxxxxxxb:9,xxxxxxxxxxxxxbxxxx:9,y:[11,13,24,25,27],yield:19,you:[0,2,20,24,26,31],your:[2,31],z:[11,24],zero_grad:[19,28],zip:[3,14,19]},titles:["Introduction","Advanced Topics","Saving","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.ast_utils</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.inspector</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.packagefinder</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.serialize</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.sourceget</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.symfinder</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.var2mod</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.exceptions</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.print_utils</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.transforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.util_transforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.utils</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.version</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.wrap</span></code>","Getting Started","Welcome to PADL\u2019s documentation!","API Documentation","Usage","Applying Transforms to Data","Combining Transforms into Pipelines","Creating Transforms","Extras","Printing and Slicing","Using PyTorch Modules with Transforms","Saving and Loading","Stages: Preprocess, Forward and Postprocess","<em>Transforms</em> and <em>Pipelines</em>"],titleterms:{"import":22,The:26,access:28,advanc:1,an:24,anyth:22,api:21,appli:[19,23,26],ast_util:5,automat:28,basic:19,between:19,block:2,build:24,combin:24,compos:24,content:[1,20,21,22],contribut:0,convert:24,creat:25,custom:2,data:[19,23],decompos:19,defin:[2,19],depend:26,devic:28,dict:28,dictionari:24,differ:24,document:[20,21],doe:2,dumptool:[4,5,6,7,8,9,10,11],exampl:24,except:[12,26],extra:26,extract:24,first:0,forward:[24,30],from:[2,22,24],gener:24,get:[0,19],group:24,handl:26,how:2,imag:24,input:19,insid:19,inspector:6,instal:[0,19],introduct:0,item:24,layer:28,licens:0,load:[19,29],loop:2,map:24,mode:26,model:[19,24],modul:[2,28],multipl:24,mutat:2,name:19,need:22,nest:2,object:2,other:2,packagefind:7,padl:[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,28],parallel:24,paramet:28,pass:[19,24],pipelin:[19,24,31],post:24,postprocess:30,pre:24,preprocess:30,print:27,print_util:13,process:24,program:0,project:19,pytorch:[2,28],resourc:[0,19],rollout:24,s:20,same:26,sampl:24,save:[2,19,28,29],scope:2,serial:[2,8],share:28,slice:27,sourceget:9,stage:30,start:[0,19],state:28,structur:19,symfind:10,target:2,tensor:24,topic:1,train:[19,24],transform:[2,14,19,23,24,25,26,28,31],us:28,usag:[19,22],util:[16,26],util_transform:15,valu:2,var2mod:11,variabl:2,version:[17,24],weight:28,welcom:20,what:2,within:2,work:2,wrap:18,you:22,your:0}})