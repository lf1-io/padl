Search.setIndex({docnames:["README","advanced","advanced/saving","apidocs/padl","apidocs/padl.dumptools","apidocs/padl.dumptools.inspector","apidocs/padl.dumptools.packagefinder","apidocs/padl.dumptools.serialize","apidocs/padl.dumptools.sourceget","apidocs/padl.dumptools.symfinder","apidocs/padl.dumptools.var2mod","apidocs/padl.exceptions","apidocs/padl.print_utils","apidocs/padl.transforms","apidocs/padl.util_transforms","apidocs/padl.utils","apidocs/padl.version","apidocs/padl.wrap","gettingstarted","index","modules","usage","usage/apply","usage/combining_transforms","usage/creating_transforms","usage/extras","usage/print_slice","usage/pytorch","usage/saving","usage/stages","usage/transform"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["README.md","advanced.md","advanced/saving.md","apidocs/padl.md","apidocs/padl.dumptools.md","apidocs/padl.dumptools.inspector.md","apidocs/padl.dumptools.packagefinder.md","apidocs/padl.dumptools.serialize.md","apidocs/padl.dumptools.sourceget.md","apidocs/padl.dumptools.symfinder.md","apidocs/padl.dumptools.var2mod.md","apidocs/padl.exceptions.md","apidocs/padl.print_utils.md","apidocs/padl.transforms.md","apidocs/padl.util_transforms.md","apidocs/padl.utils.md","apidocs/padl.version.md","apidocs/padl.wrap.md","gettingstarted.md","index.md","modules.md","usage.md","usage/apply.md","usage/combining_transforms.md","usage/creating_transforms.md","usage/extras.md","usage/print_slice.md","usage/pytorch.md","usage/saving.md","usage/stages.md","usage/transform.md"],objects:{"":{padl:[3,0,0,"-"]},"padl.dumptools":{inspector:[5,0,0,"-"],packagefinder:[6,0,0,"-"],serialize:[7,0,0,"-"],sourceget:[8,0,0,"-"],symfinder:[9,0,0,"-"],var2mod:[10,0,0,"-"]},"padl.dumptools.inspector":{CallInfo:[5,1,1,""],caller_frame:[5,4,1,""],caller_module:[5,4,1,""],get_segment_from_frame:[5,4,1,""],get_statement:[5,4,1,""],get_surrounding_block:[5,4,1,""],non_init_caller_frameinfo:[5,4,1,""],outer_caller_frameinfo:[5,4,1,""],trace_this:[5,4,1,""]},"padl.dumptools.inspector.CallInfo":{module:[5,3,1,""]},"padl.dumptools.packagefinder":{dump_packages_versions:[6,4,1,""],get_packages:[6,4,1,""],get_version:[6,4,1,""]},"padl.dumptools.serialize":{Serializer:[7,1,1,""],json_serializer:[7,4,1,""],load_json:[7,4,1,""],save_json:[7,4,1,""],value:[7,4,1,""]},"padl.dumptools.serialize.Serializer":{save:[7,5,1,""],save_all:[7,5,1,""],varname:[7,3,1,""]},"padl.dumptools.sourceget":{ReplaceString:[8,1,1,""],ReplaceStrings:[8,1,1,""],cut:[8,4,1,""],get_module_source:[8,4,1,""],get_source:[8,4,1,""],original:[8,4,1,""],put_into_cache:[8,4,1,""]},"padl.dumptools.sourceget.ReplaceString":{cut:[8,5,1,""]},"padl.dumptools.sourceget.ReplaceStrings":{cut:[8,5,1,""]},"padl.dumptools.symfinder":{NameNotFound:[9,6,1,""],Scope:[9,1,1,""],ScopedName:[9,1,1,""],find:[9,4,1,""],find_in_function_def:[9,4,1,""],find_in_ipython:[9,4,1,""],find_in_module:[9,4,1,""],find_in_scope:[9,4,1,""],find_in_source:[9,4,1,""],replace_star_imports:[9,4,1,""],split_call:[9,4,1,""]},"padl.dumptools.symfinder.Scope":{empty:[9,5,1,""],from_level:[9,5,1,""],from_source:[9,5,1,""],global_:[9,5,1,""],is_global:[9,5,1,""],module_name:[9,3,1,""],toplevel:[9,5,1,""],unscoped:[9,5,1,""],up:[9,5,1,""]},"padl.dumptools.var2mod":{CodeNode:[10,1,1,""],Finder:[10,1,1,""],Vars:[10,1,1,""],find_globals:[10,4,1,""],increment_same_name_var:[10,4,1,""],unscope_graph:[10,4,1,""]},"padl.dumptools.var2mod.Finder":{find:[10,5,1,""],generic_visit:[10,5,1,""]},"padl.dumptools.var2mod.Vars":{globals:[10,7,1,""],locals:[10,7,1,""]},"padl.exceptions":{WrongDeviceError:[11,6,1,""]},"padl.print_utils":{combine_multi_line_strings:[12,4,1,""],create_arrow:[12,4,1,""],create_reverse_arrow:[12,4,1,""],format_argument:[12,4,1,""],make_bold:[12,4,1,""],make_green:[12,4,1,""],visible_len:[12,4,1,""]},"padl.transforms":{AtomicTransform:[13,1,1,""],Batchify:[13,1,1,""],BuiltinTransform:[13,1,1,""],ClassTransform:[13,1,1,""],Compose:[13,1,1,""],CompoundTransform:[13,1,1,""],FunctionTransform:[13,1,1,""],Identity:[13,1,1,""],Map:[13,1,1,""],Parallel:[13,1,1,""],Rollout:[13,1,1,""],TorchModuleTransform:[13,1,1,""],Transform:[13,1,1,""],Unbatchify:[13,1,1,""],group:[13,4,1,""],load:[13,4,1,""],save:[13,4,1,""]},"padl.transforms.ClassTransform":{source:[13,3,1,""]},"padl.transforms.CompoundTransform":{grouped:[13,5,1,""],pd_forward_device_check:[13,5,1,""],pd_to:[13,5,1,""]},"padl.transforms.FunctionTransform":{source:[13,3,1,""]},"padl.transforms.TorchModuleTransform":{post_load:[13,5,1,""],pre_save:[13,5,1,""]},"padl.transforms.Transform":{eval_apply:[13,5,1,""],infer_apply:[13,5,1,""],pd_call_transform:[13,5,1,""],pd_device:[13,3,1,""],pd_forward:[13,3,1,""],pd_forward_device_check:[13,5,1,""],pd_get_loader:[13,5,1,""],pd_layers:[13,3,1,""],pd_name:[13,3,1,""],pd_parameters:[13,5,1,""],pd_post_load:[13,5,1,""],pd_postprocess:[13,3,1,""],pd_pre_save:[13,5,1,""],pd_preprocess:[13,3,1,""],pd_save:[13,5,1,""],pd_set_mode:[13,5,1,""],pd_to:[13,5,1,""],pd_varname:[13,5,1,""],pd_zip_save:[13,5,1,""],train_apply:[13,5,1,""]},"padl.util_transforms":{IfEval:[14,1,1,""],IfInMode:[14,1,1,""],IfInfer:[14,1,1,""],IfTrain:[14,1,1,""],Try:[14,1,1,""]},"padl.utils":{same:[15,7,1,""]},"padl.wrap":{PatchedModule:[17,1,1,""],transform:[17,4,1,""]},padl:{Batchify:[3,1,1,""],Identity:[3,1,1,""],IfEval:[3,1,1,""],IfInMode:[3,1,1,""],IfInfer:[3,1,1,""],IfTrain:[3,1,1,""],Unbatchify:[3,1,1,""],batch:[3,2,1,""],dumptools:[4,0,0,"-"],exceptions:[11,0,0,"-"],group:[3,4,1,""],identity:[3,2,1,""],load:[3,4,1,""],print_utils:[12,0,0,"-"],save:[3,4,1,""],transform:[3,4,1,""],transforms:[13,0,0,"-"],unbatch:[3,2,1,""],util_transforms:[14,0,0,"-"],utils:[15,0,0,"-"],value:[3,4,1,""],version:[16,0,0,"-"],wrap:[17,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","data","Python data"],"3":["py","property","Python property"],"4":["py","function","Python function"],"5":["py","method","Python method"],"6":["py","exception","Python exception"],"7":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:data","3":"py:property","4":"py:function","5":"py:method","6":"py:exception","7":"py:attribute"},terms:{"0":[0,3,5,8,9,10,13,18,25,26],"02_nlp_exampl":18,"1":[0,2,3,8,9,10,13,14,18,23,25,26,27],"10":[0,3,14,18,24],"100":[2,3,14,23,24,25,26],"1000":24,"11":[3,5,14],"123":[5,9],"13":8,"2":[0,3,8,9,10,13,18,23,25,26],"20":0,"200":25,"244":23,"3":[0,3,5,9,13,18,23,25],"300":25,"4":[10,25],"5":8,"512":18,"64":18,"8":[0,18],"9":[0,8,18],"abstract":[0,13,18,19,30],"case":[2,8,9,23],"catch":[14,21],"class":[2,3,5,7,8,9,10,13,14,17,18,24,29],"const":2,"default":[3,5,6,7,9,13,14,23],"do":[2,3,11,13,21,22,24],"final":[14,18,25],"function":[0,2,3,5,7,9,10,13,17,18,23,24,30],"import":[0,2,3,9,10,13,14,17,18,19,23,24,25,28,29],"int":[5,8,9,10,12,13],"new":[8,9],"return":[0,2,5,6,7,8,9,13,18,22,23,24,25,26,29],"static":13,"super":18,"true":[3,8,9,10,12,13,17,18,23,26],"try":[3,5,8,13,14,17,21,25],"var":10,A:[5,8,9,13,18,30],As:[2,29],By:23,For:[2,18,23,24,25],If:[0,2,3,7,8,9,12,13,14,18],In:[8,9,18,23,24],It:[2,5,9,13,22,24,27],No:14,One:23,The:[2,3,5,7,8,9,12,13,14,18,21,23,29,30],These:[2,13,18],To:[2,3,8,13,18,22,23,29],With:30,_:[2,18],____________:12,__call__:[2,18,24],__file__:2,__getitem__:18,__init__:[2,5,18,24],__main__:9,_ast:[9,10],_pd_main:2,_thingfind:9,_trace_thi:5,about:[5,13,27,30],abov:[18,23],absolut:2,accept:13,access:[9,18,26,29],ad:[3,9,13,22,30],adam:[18,27],adapt:27,add:[2,3,8,13,24,25,26],add_n:2,addconst:2,addfirst:25,addit:8,advanc:19,after:[2,3,13,27],again:2,ai:0,airplan:22,alia:10,all:[0,2,5,7,10,11,13,17,18,21,23,27,28],allow:[0,8,9,18,23,25],along:27,also:[2,3,13,18,22,24,25],an:[0,2,5,8,9,10,12,13,18,22,25,27,30],analog:25,ani:[7,12,13,14,18],anoth:12,anyth:[2,19],apach:0,api:19,append:[2,7,18],appli:[3,12,13,14,19,21,23,29],ar:[2,3,7,9,10,13,18,21,23,24,27,28,29],arbitrari:[0,25],arg:[2,5,13,18,26],argument:[2,5,9,12,13],aris:23,arrai:[2,25],arrow:12,ascii:12,assign:[9,13],ast:[6,9,10],ast_nod:10,atom:13,atomictransform:13,attribut:[5,8,15],augment:[23,25,27,29],automat:[2,3,13,22],avail:[5,22],awar:2,axxxxxxxxxxxxxxxxxxxx:8,b:[3,8,9,10,13,14,18],back:14,backward:[18,27],bar:[13,23],base:[0,7,13],batch:[0,3,13,18,21,22,23,27,29,30],batch_first:18,batch_siz:[18,27],batchifi:[3,9,13,21],baz:23,becom:[5,9,18,23,24],been:13,befor:[2,5,13,22],being:[3,9,13,25],below:10,berlin:0,between:23,big:2,blob:0,block:[5,18],bodi:[2,5,9],boilerpl:30,bold:12,bool:[3,5,8,9,13],both:[2,8,27],branch:18,build:[0,2,9,30],build_my_transform:2,builder:0,built:[18,27],builtintransform:13,c:[3,13],cach:8,call:[3,5,7,9,10,13,14,24,29],call_info:13,call_sourc:9,callabl:[3,5,7,13,17,18,24],caller:5,caller_fram:5,caller_modul:5,callinfo:[5,13],calling_scop:9,can:[2,5,8,9,13,14,18,22,23,24,25,26,27,28,29,30],cannot:2,capabl:27,captur:15,carri:14,cat:[18,22],catch_transform:14,caveat:2,cell:8,central:30,chang:[2,5,14],charact:12,check:[13,18],checkpoint:28,chief:18,child:[11,13],child_transform:11,children:13,citizen:18,classifi:[23,29],classmethod:[7,9],classtransform:13,claus:14,clean:[2,18],close:25,co:18,code:[0,2,3,5,6,7,8,9,13,27,30],code_of_conduct:0,codegraph:7,codenod:10,col:[5,8],collect:[8,13,21],column:5,com:[0,18],combin:[12,13,18,19,21,24,30],combine_multi_line_str:12,come:[17,24],comment:5,common:23,complet:[5,14],complex:[2,13,26,30],compon:[0,18,23],compos:[13,18,26,29],composit:[18,23],compound:[13,21],compoundtransform:13,comprehens:18,compress:[3,13],comput:18,concat_low:25,concis:[18,21],condit:[14,21],condition:23,conduct:0,conflict:[2,10],consid:2,consist:2,constant:[2,8],construct:13,contain:[2,5,7,8,9,11,12,13,27,28],context:[5,13],continu:[23,29],contrast:13,contribut:19,conveni:27,convert:[0,9,24,27,29],coor:12,coordin:12,correct:13,correspond:[9,29],cosin:18,could:[2,9,23],counter:10,cpu:[3,13],creat:[2,5,7,9,10,12,13,19,21,22,23,27,28,30],create_arrow:12,create_reverse_arrow:12,crop:23,cuda:[13,27],current:[0,2,9,18],custom:[13,30],cut:8,data:[2,13,19,21,29,30],dataload:[13,22,30],datapoint:[13,23],dataset:2,debug:30,decor:[3,13,17,18,24],deep:[0,2,18,19,23,29,30],def:[0,2,5,13,18,24,25,26,29],def_sourc:9,defin:[7,9,21,28,29],definit:18,depend:[2,28],detail:[5,13,18,23],determin:[3,5,8,9,13,17],develop:0,devic:[3,11,13],di:30,dict:[8,18],dictionari:[2,13,18],differ:[8,22,29],dim:[3,13],dimens:[3,13,22,30],directli:13,directori:2,disabl:[12,22],document:[5,13],doe:[13,22],doesn:[3,14],dog:[18,22],don:[2,3,5,12,13,17],done:[13,22],down:2,drop:[5,9],drop_n:[5,9],dump:[6,7,13],dump_packages_vers:6,dumptool:[13,19,20],dure:[3,13,23,25],dynam:18,e:[3,9,13,18],each:[2,13,18,22,29],easili:30,ecosystem:0,either:[2,8],eleg:30,element:[13,18,23],els:[3,14],else_:[3,14],else_transform:14,emb:18,embed:18,emit:18,empti:[5,9],enabl:30,end:[3,12,13],enter:5,entir:24,entiti:9,entri:2,enumer:2,eos_valu:18,equival:25,error:14,escap:12,eval:[3,13,14,21,22,25],eval_appli:[3,13,14,18,22,25],evalu:29,even:23,event:5,everi:12,everyth:[21,23,24],exactli:23,exampl:[2,3,5,8,9,10,12,13,14,17,18,25,29],except:[9,14,19,20,21],execut:[2,5,23],exist:[3,10,13],expect:[2,3,8,13,22],explicit:[8,10],explicitli:13,extens:[3,13],extra:[14,19,21,27],extract:5,f1:13,f2:13,f:[2,3,8,9,10,13],factori:15,fail:14,fall:14,fals:[3,5,9,12,13,17],few:2,field:10,file:[2,3,8,13,25,28],file_suffix:7,fileexistserror:[3,13],filenam:[2,8,13,25],filter_builtin:10,finally_transform:14,find:[6,9,10,18],find_glob:10,find_in_function_def:9,find_in_ipython:9,find_in_modul:9,find_in_scop:9,find_in_sourc:9,finder:10,finish_right:12,first:[3,5,8,12,13,18,23],flat:23,flatten:[3,13,23],flexibl:30,folder:[2,3,13,28],follow:[2,5,21],foo:[13,23],force_overwrit:[3,13],form:[9,13,24,30],formal:30,format:[6,12,29,30],format_argu:12,forward:[0,13,18,19,21,22,24],forward_pass:18,found:[5,9],frame:[5,8],frameinfo:5,from:[0,2,3,5,8,9,13,14,18,19,24,25,27,28,29],from_col:8,from_level:9,from_lin:8,from_sourc:9,front:5,full:[0,18],functiontransform:13,fundament:13,futur:2,g:[9,13,18],gener:[13,22],generic_visit:10,germani:0,get:[2,5,6,8,9,12,13,15,19,21,23,25],get_module_sourc:8,get_packag:6,get_segment_from_fram:5,get_sourc:8,get_stat:5,get_surrounding_block:5,get_vers:6,getitem:[5,26],github:[0,18],give:8,given:[3,5,6,8,9,10,13],global:[2,9,10],global_:9,globals_:10,go:[10,18],good:13,gotten:8,gpu:[13,22,29,30],gradient:[13,18,22,30],graph:[10,18],great:0,green:12,group:[3,13,21],gru:18,ha:[2,8,13,18,23,27,29],hacki:5,happen:[14,29],have:[0,13,27],head:23,hello:25,help:[13,18],helper:[3,7],here:[2,5,18],hidden:24,hierarchi:9,histori:9,hood:22,how:[9,13,22,23,24,26,28,29,30],howev:2,http:[0,18],i:[2,3,9,13],id:8,ident:[3,13,14,21,26],identifi:9,if_:[3,14],ifev:[3,14,21,25],iff:9,ifinf:[3,14,21,25],ifinmod:[3,14],iftrain:[3,14,18,21,23,25],ignore_scop:[3,5,13,17],ii:[3,14],imag:[12,29],implement:[13,18,24],includ:[2,8,13,18,27,28],increment:10,increment_same_name_var:10,indent:5,index:[13,18],indic:[9,13],individu:[18,23],infer:[3,13,14,21,22,25,27,29],infer_appli:[3,13,14,18,22,25],infer_pipelin:27,inform:[5,13],initi:[2,13,27],inlin:[5,13,18],innov:0,input:[3,8,12,13,22,23,25],insert:8,insid:14,inspect:[5,8,13,30],inspector:[13,19,20],instanc:[2,18,23,24,27],instanti:18,instead:[2,3,5,13,17],instruct:18,interact:[2,28,30],interest:0,intermedi:18,intern:22,introduct:19,invers:[3,13],io:[0,18],ipelin:27,ipynb:18,ipython:[2,8,9,26],is_glob:9,isinst:[8,10,17,18],issu:[0,18],item:[0,3,13,15,22,25,29],iter:[13,22,27],its:[9,13],itself:5,jpg:22,json:[2,7],json_seri:7,just:[3,13],keep:23,kei:8,keyword:[0,13],kic:18,know:21,kwarg:[13,18],l1:24,l2:24,lab:0,lambda:[3,13,14,18,23,24,25],larg:[2,18],larger:[5,18],largest:5,last:[2,8,12,13],layer1:27,layer2:27,layer:[0,13,18],learn:[0,18,19,22,23,24,27,28,29,30],left_shift:18,len:[2,18],length:[12,23],level:[2,5,9,10,23],lf1:[0,18],licens:19,like:[2,18,22,29],limit:2,lin:24,line:[5,8,9,12],linear:[0,18,24],lineno:[5,9],link:18,list:[2,3,6,8,9,10,12,13,14,22,25,27,28],liter:[3,5,13,14],lm:18,load:[2,3,7,13,19,21,22,23,27,29],load_data:2,load_funct:7,load_imag:[23,29],load_json:7,loader:[7,13],loc:8,local:10,locat:[5,8],log10:24,look:[0,2,5,9,18],lookup:[27,29],loop:[9,27],loos:18,loss:[0,18,23,27],lower:25,lower_cas:18,lr:[18,27],macro:23,made:[5,13],mai:[0,18],main:[0,9,11,18],maintain:8,make:[0,12,21,27],make_bold:12,make_green:12,man:18,manual:18,map:[13,18],mark:[3,7,13],mat:18,match:[3,5,11,14],math:24,md:0,mean:[2,22,23],messi:2,method:[5,13,18,21,25],might:[2,18,25],minim:30,minus100:24,minusx:[24,25,26],mlp:24,mml:24,mode:[3,13,14,22],model:[0,2,13,27,29],model_pass:23,modifi:8,modul:[3,5,6,7,8,9,13,17,18,19,21,23,24,26,28,29],module_nam:[5,9],more:[2,5,23,30],most:[23,29],mother:11,mother_transform:11,move:[3,13,23],much:30,multi:[5,12,30],multipl:[2,13,18,22,24,27],multiprocess:13,must:[2,5,13,23],my:2,my_classifi:[22,29],my_classifier_transform:[23,29],my_pipelin:[0,2,27,28],myfil:2,myfunct:5,mykei:8,myload:2,mypipelin:28,mypytorchlay:27,mysav:2,mytransform:2,n:[5,9,10,13],n_final_row:12,n_in:24,n_initial_row:12,n_out:24,n_to_add:2,n_word:18,name:[2,3,5,6,9,10,13,14,29],namenotfound:9,natur:22,need:[2,7,13,19],nest:[9,26,27],next:[18,22,23,24,26,27,28,29,30],nextmodul:5,nn:[0,13,18,24],node:[6,9,10],nodetyp:10,non:[8,10],non_init_caller_frameinfo:5,none:[3,5,7,9,10,13,14,17],nonetyp:10,normal:[2,8,13,18],not_found:2,note:[2,5],notebook:[2,18,30],noth:[3,13,21],now:2,np:[2,17,24,25],npy:2,nth:23,num_work:[18,27],number:[5,9,10,12],numpi:[2,17,18,24],o:[18,27],object:[5,8,9,13,18,25],obtain:18,occur:14,offset:5,often:[18,25],one:[2,3,9,12,13,14,18,23],onli:[2,18,23,25],open:[2,23,25],open_fil:25,oper:[18,23],optim:[18,27],option:[3,5,7,13,14],order:13,ordereddict:13,origin:[5,8],other:[2,13,14],otherwis:[3,13,14],out:[5,8,9,10,14,17,23,24],outer:5,outer_caller_frameinfo:5,output:[3,13,14,18,22,23,29],over:[12,13,18,23,27],overlap:8,overrid:13,overwrit:[3,13],own:2,packag:[0,2,6,28],packagefind:[19,20],pad:12,padl:[2,18,20,23,24,25,26,28,29,30],padl_value_0:2,page:21,parallel:[3,13,18],param:13,paramet:[2,3,5,6,7,8,9,11,12,13,14,17,18,28],parent:[2,17],parenthes:23,parrallel:23,pars:10,part:[8,13,18,21,22,29],parti:2,particular:24,pass:[0,2,3,13],patch:17,patchedmodul:17,path:[2,3,7,13,23,29],pathlib:[2,3,7,13],pd_:18,pd_call_transform:13,pd_devic:13,pd_forwad:29,pd_forward:[13,22,29],pd_forward_device_check:13,pd_get_load:13,pd_group:13,pd_layer:[13,27],pd_name:[13,14],pd_np:17,pd_paramet:[13,18,27],pd_post_load:13,pd_postprocess:[13,29],pd_pre_sav:13,pd_preprocess:[13,29],pd_save:[13,18],pd_set_mod:13,pd_to:[13,27],pd_varnam:13,pd_zip_sav:13,perform:[3,13,14],pick:2,piec:9,pil:23,piltotensor:23,pip:[0,18],pipelin:[0,19,21,24,26,27,28,29],plan:18,pleas:0,pluson:2,plustwo:2,point:18,posit:8,position:18,possibl:[2,5,13,18,24,27],post:[0,18,29],post_load:13,postprocess:[3,13,18,19,21,22,23],potenti:[2,5,18],power:23,pre:[0,8,18,29],pre_sav:13,preced:12,predict:[18,23,27],prepar:0,prepend:10,preprocess:[3,13,18,19,21,22,23,27,30],preprocess_imag:23,pretti:26,prevent:[2,3,10,13],previous:2,primit:18,print:[5,12,13,18,19,21,28],print_util:[19,20],process:[0,2,18,22,29,30],progress:13,propag:13,properti:[5,7,9,13],provid:[2,23],pt:2,put:8,put_into_cach:8,py:[2,28],pypi:2,python:[0,2,5,9,18,23,24,28],pytorch:[0,13,18,19,21,22,23,24,26,28,29],rais:[3,5,13],rand:17,random:17,randomresizedcrop:23,randomrot:23,rang:[2,18],rather:[2,5,13],re:[0,23],read:[2,21,22,25,26,30],read_from_path:25,readabl:29,recreat:2,recurs:[3,13,18],refer:[18,21],referenc:18,reflex:18,regardless:14,releas:18,relu:24,rememb:27,remov:[3,5,13,22],renam:[2,10],repeat:[2,23],repl:8,replac:[8,9],replace_cach:8,replace_star_import:9,replacestr:8,repres:[9,18],reproduc:30,requir:[2,5],reshap:29,resiz:23,resnet18:[23,29],respect:[3,13,25],respons:[2,29],rest:5,result:[2,3,5,8,13,23],return_loc:5,return_parti:9,right_shift:18,rnn:18,roll:23,rollout:[3,13,18],row:5,rstring:8,s:[3,5,6,7,8,9,13,14,18,21,24,27,28,29],same:[3,5,8,10,13,15,21,23,27],sat:18,satement:5,save:[1,3,7,13,19,21,22,30],save_al:7,save_funct:7,save_json:7,saver:7,scalar:18,schemat:18,scope:[3,5,9,10,13,17],scoped_nam:10,scopednam:[9,10],scopelist:9,scopemap:[7,10],search:[9,13],second:[2,3,13,23],section:[8,22,23,24,26,27,28,29,30],see:[0,2,3,5,13,18],segement_typ:5,segment:[5,9],segment_typ:5,self:[2,9,13,18,24],send:[13,22,27,30],sequenc:[12,13,23],seri:13,serial:[3,13,19,20],session:[2,28],set:[2,7,10,13],settrac:5,sever:18,sh:18,should:[2,3,5,7,13],similar:2,simpli:[2,18],sin:24,singl:[0,13,18,22,23,29],slice:[18,19,21,28,30],so:[9,18],some:[2,23,27],somemodul:10,someobject:2,someth:[2,25],sometim:2,soon:5,sourc:[5,8,9,10,13,18],sourceget:[19,20],space:[5,12],special:[23,25,29],specifi:[3,8,9,13],split:[2,9,18],split_cal:9,split_str:18,stack:5,stage:[18,19,21,22,23],stand:29,standard:[0,18],star:[2,9],start:[3,9,12,13,19,21],start_left:12,state_dict:27,statement:[2,5,9,14],step:[13,18,23,27],step_1:18,stop:30,store:[2,3,7,27,28],str:[2,3,5,7,8,9,10,12,13,14],straightforward:18,string:[6,7,8,9,12,13],sub:[8,13,23,26],subclass:9,subnod:10,substr:8,subval:2,successfulli:14,suffix:2,suggest:29,superimpos:12,support:[0,2,18],surround:[5,9],sy:5,symbol:[2,9],symfind:[5,10,19,20],t1:[13,23,26],t2:[13,23,26],t3:[13,23,26],t4:26,t5:26,t:[2,3,5,12,13,14,17,18,23,25],take:[12,23],taken:[8,24],tansform:29,target:[5,23],target_mod:[3,14],tensor:[0,3,13,18,22],term:18,test:18,text:[0,2],textcorpu:2,th:18,than:2,them:[2,3,13,21,23,27],thereof:[3,13],thi:[2,3,5,8,9,12,13,14,18,22,23,24,26,27,28],thing:[2,3,7,8,9],third:2,those:[2,10,27,29],thought:18,three:[22,23],through:[10,18,23],thu:23,time:2,to_col:8,to_integ:18,to_lin:8,to_tensor:18,toarrai:2,togeth:[18,26],tointeg:18,token:18,too:[24,26,28],top:[9,10,12],topic:19,topk:0,toplevel:[3,5,9,13,17],topolog:0,torch:[0,13,18,24,27],torchmoduletransform:[13,18],torchvis:[18,23,29],totensor:[23,29],trace:5,trace_thi:5,tracefunc:5,track:2,train:[3,13,14,22,25,27,29],train_appli:[13,18,22,27],train_data:[18,27],train_model:27,train_pipelin:[18,27],training_pipelin:23,transform:[0,3,9,11,14,15,17,19,20,21,26,28,29],transform_1:18,transform_2:18,transpar:[12,30],tree:[9,10,18],tupl:[2,3,5,9,10,13,14,18,23],turn:13,tvt:23,two:2,txt:[2,28],type:[5,10],typic:[13,18],unbatch:[0,3,18,21,23,29],unbatchifi:[3,13],under:[0,22],undo:2,unind:5,union:[3,10,13,14],uniqu:13,unk:18,unscop:9,unscope_graph:10,unsqueez:[3,13],up:[2,9],upcom:18,updat:22,us:[0,2,3,5,6,7,8,9,10,12,13,14,17,18,19,21,22,23,24,25,26,29,30],usag:19,use_replace_cach:8,user:0,usual:29,util:[5,6,12,13,19,20,23,27],util_transform:[19,20],v:12,val:[2,3,7],valu:[3,7,8,9,12,18,21],valueerror:5,var2mod:[19,20],var_nam:9,variabl:[9,10,13],variou:[8,9],varnam:[7,9],verbos:13,veri:2,version:[0,2,6,8,9,10,13,19,20,28],via:[2,13,18,26,27,28,29],view:23,visibl:12,visible_len:12,visitor:10,wa:[0,5,9,13],wai:9,want:[2,25],we:[2,18,29],well:9,were:[2,9],what:[13,18,25,29],whatev:13,when:[2,3,13,14,24,27,28],where:[5,8,9,10,12],wherea:29,which:[2,5,8,9,13,18,23,28],whole:[13,30],whose:[3,13],wide:5,within:[5,8,9,13,27],without:30,word:[2,18],word_dropout:18,word_index:2,wordindex:2,work:[26,27,30],worker:22,workflow:[18,29,30],worri:30,would:[2,8,9,18,23],wrap:[2,3,13,19,20,24],wrappe:[3,17],wrapper:[3,17,24],write:[0,2,3,13],written:9,wrongdeviceerror:11,x1:13,x2:13,x3:13,x:[0,2,3,8,10,12,13,14,18,23,24,25,26,29],xxxx:8,xxxxxaxxxxxxxxxxxxxxxxxxxx:8,xxxxxxxxxxx:8,xxxxxxxxxxxxxb:8,xxxxxxxxxxxxxbxxxx:8,y:[10,12,23,24,26],yield:18,you:[0,2,19,23,25,30],your:[2,30],z:23,zero_grad:[18,27],zip:[3,13,18]},titles:["Introduction","Advanced Topics","Saving","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.inspector</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.packagefinder</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.serialize</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.sourceget</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.symfinder</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.dumptools.var2mod</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.exceptions</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.print_utils</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.transforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.util_transforms</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.utils</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.version</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">padl.wrap</span></code>","Getting Started","Welcome to PADL\u2019s documentation!","API Documentation","Usage","Applying Transforms to Data","Combining Transforms into Pipelines","Creating Transforms","Extras","Printing and Slicing","Using PyTorch Modules with Transforms","Saving and Loading","Stages: Preprocess, Forward and Postprocess","<em>Transforms</em> and <em>Pipelines</em>"],titleterms:{"import":21,The:25,access:27,advanc:1,an:23,anyth:21,api:20,appli:[18,22,25],automat:27,basic:18,between:18,block:2,build:23,combin:23,compos:23,content:[1,19,20,21],contribut:0,convert:23,creat:24,custom:2,data:[18,22],decompos:18,defin:[2,18],depend:25,devic:27,dict:27,dictionari:23,differ:23,document:[19,20],doe:2,dumptool:[4,5,6,7,8,9,10],exampl:23,except:[11,25],extra:25,extract:23,first:0,forward:[23,29],from:[21,23],gener:23,get:[0,18],group:23,handl:25,how:2,imag:23,input:18,insid:18,inspector:5,instal:[0,18],introduct:0,item:23,layer:27,licens:0,load:[18,28],loop:2,map:23,mode:25,model:[18,23],modul:[2,27],multipl:23,mutat:2,name:18,need:21,nest:2,object:2,packagefind:6,padl:[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,27],parallel:23,paramet:27,pass:[18,23],pipelin:[18,23,30],post:23,postprocess:29,pre:23,preprocess:29,print:26,print_util:12,process:23,program:0,project:18,pytorch:[2,27],resourc:18,rollout:23,s:19,same:25,sampl:23,save:[2,18,27,28],scope:2,serial:[2,7],share:27,slice:26,sourceget:8,stage:29,start:[0,18],state:27,structur:18,symfind:9,target:2,tensor:23,topic:1,train:[18,23],transform:[2,13,18,22,23,24,25,27,30],us:27,usag:[18,21],util:[15,25],util_transform:14,valu:2,var2mod:10,variabl:2,version:[16,23],weight:27,welcom:19,what:2,within:2,work:2,wrap:17,you:21,your:0}})