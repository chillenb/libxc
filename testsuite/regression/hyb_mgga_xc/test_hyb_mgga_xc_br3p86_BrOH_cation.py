
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_br3p86_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.490635470410513e+01, -1.490638374666013e+01, -1.490656388138824e+01, -1.490613111285669e+01, -1.490635063586560e+01, -1.490635063586560e+01, -2.788952817174019e+00, -2.788919374968669e+00, -2.788196216526361e+00, -2.789896676397335e+00, -2.788968513672550e+00, -2.788968513672550e+00, -6.046402016056952e-01, -6.045894776824433e-01, -6.043102272917688e-01, -6.068244895148778e-01, -6.059620859712952e-01, -6.059620859712952e-01, -1.915046086632167e-01, -1.928278116737069e-01, -6.987910359485004e-01, -1.636524422301738e-01, -1.814326018823731e-01, -1.814326018823461e-01, -5.753999082500565e-02, -5.785751557745688e-02, -1.038871891346229e-01, -5.115072905528206e-02, -5.299087046342484e-02, -5.299087046339831e-02, -6.401452315204496e+00, -5.300754365538676e+00, -3.164288508468709e+00, -7.551401954004536e+00, -3.383463316095804e+00, -4.594069861665564e+00, -1.199842782890448e+00, -2.004775870625051e+00, -1.802545861858597e+00, -1.809029455498303e+00, -1.809336903022394e+00, -1.668692221463500e+00, -9.243298373562273e-01, -7.765685669655719e-01, -8.459287713255064e-01, -5.075627778534272e-01, -5.452102288596877e-01, -5.726828931785520e-01, -5.343147878931877e-02, -3.065540786850591e-01, -4.976959355348468e-02, -1.631172827741137e+00, -1.339852155642570e-01, -1.196053241058680e-01, -9.845995111952453e-02, -2.057630266126237e+00, -3.342077484004011e-03, -3.757359581525671e-02, -5.855850883973612e-02, -3.945118458021209e-03, -2.040539530230275e+01, -4.044219555521189e-01, -4.064254752830954e-01, -3.194752072331906e+00, -4.072777751604716e-01, -6.004729327868888e+01, -4.015414019957434e-01, -4.522887610904874e-01, -7.464583480228286e-01, -3.870138179126551e-01, -5.147082694138763e-01, -9.448099130357968e-01, -1.068789679730269e+00, -1.386580248522536e-01, -1.712715397794974e-01, -2.376618640498863e-01, -3.509827813482339e-01, -3.241267977092216e-01, -4.255504867063937e-01, -3.027860836205182e-02, -1.166007480315324e-01, -3.170170553924697e-01, -4.369875491205322e-02, -1.068537388121648e-01, -1.007000494043580e-02, -1.221815763810397e-03, -2.491743386047763e-03, -4.259019937939444e-02, -3.673106464893773e-03, -3.673105733373558e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_br3p86_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.178368373063534e+01, -2.178365085484421e+01, -2.178375845037611e+01, -2.178370546876666e+01, -2.178402648221918e+01, -2.178407943260389e+01, -2.178317803060884e+01, -2.178300306834971e+01, -2.178372451286097e+01, -2.178345567297056e+01, -2.178372451286097e+01, -2.178345567297056e+01, -3.607338940467163e+00, -3.607611847880605e+00, -3.607400287769729e+00, -3.607686767251748e+00, -3.609018164103691e+00, -3.609400302797065e+00, -3.606793483441050e+00, -3.607193873220233e+00, -3.606645997233949e+00, -3.608640466986245e+00, -3.606645997233949e+00, -3.608640466986245e+00, -7.430297091166832e-01, -7.464117347903307e-01, -7.415438052224539e-01, -7.456675797946946e-01, -7.195671230981380e-01, -7.147470048552675e-01, -7.211914545288381e-01, -7.226807951604904e-01, -7.466570954187843e-01, -6.985304441676677e-01, -7.466570954187843e-01, -6.985304441676677e-01, -2.155478882589150e-01, -2.196564472611467e-01, -2.185305356299382e-01, -2.233085707934403e-01, -8.507439244774812e-01, -8.759356552857350e-01, -1.634908408343708e-01, -1.643882842233175e-01, -2.003282088375732e-01, -1.607565708454496e-01, -2.003282088376504e-01, -1.607565708454496e-01, -2.873605585330512e-02, -2.890056077555564e-02, -2.926712214730686e-02, -2.942180176763478e-02, -5.211238592901724e-02, -5.250151126615454e-02, -2.333006691718498e-02, -2.339514900630646e-02, -2.589651962070507e-02, -2.353582494046533e-02, -2.589651962076496e-02, -2.353582494046521e-02, -6.512229016034388e+00, -6.323219723211608e+00, -5.824797652452403e+00, -5.819004616091634e+00, -4.257835553039210e+00, -4.253462033467242e+00, -7.080939386059652e+00, -7.134155608572063e+00, -4.864163222027029e+00, -4.866236325737110e+00, -5.543633520258461e+00, -5.516555570720215e+00, -1.609845739663127e+00, -1.610248345180272e+00, -2.133001307580821e+00, -2.132342049682737e+00, -1.975486111312968e+00, -1.979437872500847e+00, -1.987326192913268e+00, -1.994072082758939e+00, -2.015199163671654e+00, -2.004985274539924e+00, -1.926525978833594e+00, -1.921446619703815e+00, -8.751848269738489e-01, -9.089882831127899e-01, -8.216743975673343e-01, -8.231980304567968e-01, -8.311068696706324e-01, -8.155456090222336e-01, -6.600907030633425e-01, -6.755537490618719e-01, -7.055128566216624e-01, -6.788707858507415e-01, -7.130245996477299e-01, -6.865501530321374e-01, -7.239788937416186e-02, -7.301315861678516e-02, -2.917799990846068e-01, -2.920314458952725e-01, -6.247188924478745e-02, -6.548684016202160e-02, -2.127818316551769e+00, -2.127025238500030e+00, -1.187103455050644e-01, -1.235173983160333e-01, -1.016353084545071e-01, -1.043853127082555e-01, -6.053865212850918e-02, -5.948433616783926e-02, -1.223881766725103e+00, -1.199031657502664e+00, -4.397545619987903e-03, -4.332367035545573e-03, -3.381694308524735e-02, -3.399009717011808e-02, -4.015124948217329e-02, -3.231619024237124e-02, -5.465731492597767e-03, -5.060996967270755e-03, -1.296664699893617e+01, -1.304640323431419e+01, -5.398286898035314e-01, -5.411778045929906e-01, -5.398165000052079e-01, -5.411602944304478e-01, -2.383711051565666e+00, -2.305785008926489e+00, -5.397795945663795e-01, -5.411134518723576e-01, -3.673523220720934e+01, -3.683191286701892e+01, -5.247883816414419e-01, -5.258732431074902e-01, -5.635329718058502e-01, -5.651662723178661e-01, -7.458789717701275e-01, -7.501260956016799e-01, -5.640909323110600e-01, -5.553253633652285e-01, -6.255287104619810e-01, -6.271843809306100e-01, -8.762816305654673e-01, -8.765032080408786e-01, -1.013332894129888e+00, -1.010114389521953e+00, -2.194037310090340e-01, -2.202296535953321e-01, -2.640448497219458e-01, -2.657037922722198e-01, -3.480727837874611e-01, -3.489782297608230e-01, -3.947285521444705e-01, -3.953135890073854e-01, -3.766507801164050e-01, -3.774207764032813e-01, -5.247459177480107e-01, -5.279849016111253e-01, -2.580606428403530e-02, -2.584290388535512e-02, -5.607945986432056e-02, -6.152259735008218e-02, -4.009773683183488e-01, -4.046437215790452e-01, -5.366368031056095e-02, -5.624006490643153e-02, -7.118801780636576e-02, -6.951773839169759e-02, -1.307791761629703e-02, -1.298116023665800e-02, -1.616671068835643e-03, -1.615652503431969e-03, -3.313765220786435e-03, -3.259942039755831e-03, -4.441870370136388e-02, -4.976785224429133e-02, -5.055945131507278e-03, -4.712802470885636e-03, -5.055944474690115e-03, -4.712799671196045e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.919932043554574e-09, -3.387840526296933e-10, -7.919938015826395e-09, -7.919753623864948e-09, -3.387830276550948e-10, -7.919808397139028e-09, -7.919363159679516e-09, -3.387723282553065e-10, -7.919198452728845e-09, -7.921376729723790e-09, -3.387876790132655e-10, -7.921764933312409e-09, -7.919818248536600e-09, -3.387805198744331e-10, -7.920871136961504e-09, -7.919818248536600e-09, -3.387805198744331e-10, -7.920871136961504e-09, -1.683276203703304e-05, 1.037044717999379e-06, -1.683808945106169e-05, -1.683287213498219e-05, 1.037529998800064e-06, -1.683859406893889e-05, -1.683989865452737e-05, 1.048624717309748e-06, -1.684295698444546e-05, -1.681816696904737e-05, 1.028136612322332e-06, -1.682221685259036e-05, -1.684109953210486e-05, 1.037467193787525e-06, -1.682763798161012e-05, -1.684109953210486e-05, 1.037467193787525e-06, -1.682763798161012e-05, -7.510609276021811e-03, 3.313310015040289e-03, -7.618467783503268e-03, -7.490340486390674e-03, 3.296728736384830e-03, -7.619118495522392e-03, -7.363928271929600e-03, 2.843760872153977e-03, -7.243429103792830e-03, -7.194651907168963e-03, 2.768602376542056e-03, -7.247297237348779e-03, -7.967421415913913e-03, 2.808341889986223e-03, -6.607911859960349e-03, -7.967421415913913e-03, 2.808341889986223e-03, -6.607911859960349e-03, -6.960533147114975e-01, 5.385217705365746e-01, -6.224586671665511e-01, -6.847039714345080e-01, 5.375589502953496e-01, -6.015049772042872e-01, -4.151553333032284e-03, 1.750321233997727e-03, -4.284015834273571e-03, -1.409924291625905e+00, 7.313879258169717e-01, -1.350317030284645e+00, -5.320832633194780e-01, 6.193525972790854e-01, -3.798729422426447e+00, -5.320832633180916e-01, 6.193525972790852e-01, -3.798729422426445e+00, -1.125599400395919e+04, -9.236377257067202e+00, -9.439903134474107e+03, -9.982700076037883e+03, -1.234339563259111e+01, -8.194650952692336e+03, -7.837911265878186e+01, -1.740268869886274e+01, -6.882848852714918e+01, -4.661311884982296e+04, -8.806378224555604e-01, -4.850813822958399e+04, -1.537723184330354e+04, -4.995338028235889e+00, -9.374386145488250e+04, -1.537723184327004e+04, -4.995338028235929e+00, -9.374386145488296e+04, -5.153956982625154e-07, 3.315475857288729e-07, -6.172268275595328e-07, -1.121092368477775e-06, 3.357956458687620e-07, -1.129900506530919e-06, 1.649056605868821e-07, 3.317192154250272e-07, 1.651052418580295e-07, -3.169314353528857e-07, 3.354690723690711e-07, -3.038309260694398e-07, -2.985159186194248e-07, 3.337081492965287e-07, -3.040678282098405e-07, -2.010145229827935e-06, 3.337081492965287e-07, -2.223629319615542e-06, 1.763551453234117e-06, 3.533122737139200e-06, 1.762476178543438e-06, -6.273801556724099e-05, 3.716816582373674e-06, -6.281962436415791e-05, -9.606477456257412e-05, 2.760321558395404e-06, -9.610614804443713e-05, -9.511340060673180e-05, 2.930513448140418e-06, -9.423299698365906e-05, -9.416136477312960e-05, 4.019478107459801e-06, -9.438184221094133e-05, -1.401935817892934e-04, 4.019478107459801e-06, -1.340076897524847e-04, 1.845357159430885e-03, 9.460702371104496e-03, 2.222237043565097e-03, 1.285650535390302e-03, 1.065077846489308e-02, 1.328381990526125e-03, 2.537884994102932e-03, 1.247641837147677e-02, 2.392567833094237e-03, -1.671540172631872e-02, 1.635197441698068e-02, -1.376139405904590e-02, -9.841981964210095e-03, 9.298098351964816e-03, -1.277408105643224e-02, -6.717865889986501e-03, 9.298098351964816e-03, -9.070378768647575e-03, 1.049403548012160e-01, 2.233523340710482e-01, 1.049035120548593e-01, -1.040303131885554e-01, 2.959700161155134e-01, -1.016023728580719e-01, -4.969945118942897e-03, -9.688797526953849e-03, -4.970301314706296e-03, -1.217117714540941e-04, 9.300735238373925e-05, -1.220290794737298e-04, -3.483434639598507e+00, 9.624475483251858e-01, -3.172055701462504e+00, -4.794581932932211e+00, 9.624475483251858e-01, -4.683301332667060e+00, -3.796895521933863e+04, -7.987734935796332e-01, -3.479995963608406e+04, -3.909345258349684e+02, -1.564432055087186e+00, -3.847814993228140e+02, -1.968924878009063e+02, -3.937849756181016e+02, -1.968924878878423e+02, -2.200859410098340e+00, -4.385382580276580e+00, -2.197985883830620e+00, -7.215113239583741e+04, -5.542034728422161e+01, -3.934981006599263e+04, -2.771017767706756e+01, -5.542034728422135e+01, -2.771019326271232e+01, 9.019072944467762e-03, 1.807155922496046e-02, 9.019383133647564e-03, 7.664376283110152e-03, 1.532875257703269e-02, 7.664376283158334e-03, 8.046827549128223e-03, 1.609365509829253e-02, 8.046827549128482e-03, 8.185424164072414e-03, 1.685927989461112e-02, 8.172111379771427e-03, 8.229069209855671e-03, 1.645813854375216e-02, 8.229069226749026e-03, 8.225019505980790e-03, 1.645813854375216e-02, 8.225075328683452e-03, 1.101271029729700e-02, 2.202542059472326e-02, 1.101271029730077e-02, -2.458544899582930e-02, 1.436187932358166e-02, -2.450992489191305e-02, 1.860951209847498e-03, 1.507010194531762e-02, 2.010549649943888e-03, 4.766975837429283e-03, 1.629349333952539e-02, 5.898755453922266e-03, -1.166683600875943e-02, 1.563119186418936e-02, -1.144036632909350e-02, 4.726954238118497e-03, 1.563119186418936e-02, 4.744783434599939e-03, 2.699486813215578e-03, 8.511561790704803e-03, 2.683225433011282e-03, 8.090625447492630e-02, 1.618366948081996e-01, 8.090665167597513e-02, 5.465060469727116e-02, 1.093083077988367e-01, 5.465067745275556e-02, 3.121098894775543e-02, 6.291190944966338e-02, 3.124275821909459e-02, -4.632152951678636e-02, 8.481727266276692e-02, -4.576747122367288e-02, -7.463308711901896e-02, 8.481727266276697e-02, -7.347675548415970e-02, -3.078908658425102e-02, 2.058541245303033e-02, -2.976522258671567e-02, -8.666058104546369e+00, -1.730406706554454e+01, -8.675034048130968e+00, -3.531737834013666e+01, -9.818879702467862e+00, -3.055716387669718e+01, -1.082856531006353e-01, 8.908301959660438e-02, -1.065811062170052e-01, -5.154835260763448e-02, -1.020539099589044e-01, -5.177071498314553e-02, -1.238093289667614e+01, -1.020539099589045e-01, -1.219185405133014e+01, -5.740866059684738e+00, -1.148164777317931e+01, -5.740967185035071e+00, -9.030692176081746e-01, -1.758918334277055e+00, -8.847352221295283e-01, -2.028438685836395e+00, -4.056765650714165e+00, -2.028451034839767e+00, -5.373372751020711e+00, -8.398296535955597e-01, -7.824079176233827e-01, -4.820874667586606e+01, -9.641703427749275e+01, -4.820908125283800e+01, -4.820859325640570e+01, -9.641703427749310e+01, -4.820866032509938e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.498431090125332e-04, -1.498426456128405e-04, -1.498411217649795e-04, -1.498412046144802e-04, -1.498386087367617e-04, -1.498365033982354e-04, -1.498609793039096e-04, -1.498650593124773e-04, -1.498417948362934e-04, -1.498564207138773e-04, -1.498417948362934e-04, -1.498564207138773e-04, -1.447693994884668e-03, -1.448079209790945e-03, -1.447737003939686e-03, -1.448146725712097e-03, -1.449030851939370e-03, -1.449375579843055e-03, -1.446639461280945e-03, -1.447038409083114e-03, -1.447680784026762e-03, -1.448182565938874e-03, -1.447680784026762e-03, -1.448182565938874e-03, -5.455776109868850e-03, -5.585867896767717e-03, -5.413425682350845e-03, -5.570314947480178e-03, -4.886005767386381e-03, -4.720084733178863e-03, -4.824859220512131e-03, -4.886085194436876e-03, -5.774821326816958e-03, -4.056955978144578e-03, -5.774821326816958e-03, -4.056955978144578e-03, -9.380467218208868e-03, -9.716360930633687e-03, -9.612252729080905e-03, -9.993805134751611e-03, -4.534864899394876e-03, -5.139719416407207e-03, -7.843158083790127e-03, -7.898032578626658e-03, -9.312529962501778e-03, -7.018075609946257e-03, -9.312529962486438e-03, -7.018075609946252e-03, -4.447533580902656e-03, -4.475075707446438e-03, -4.537621452244662e-03, -4.578542531192107e-03, -5.448411248615187e-03, -5.536764401758736e-03, -4.002379776141124e-03, -3.960336820642341e-03, -4.314747775753964e-03, -4.847971238483982e-03, -4.314747775744565e-03, -4.847971238484009e-03, -1.908087231137432e-04, -2.191760093329318e-04, -3.612733581208704e-04, -3.634738504342098e-04, -2.672281205683553e-07, -2.111650010439544e-07, -1.358341977153134e-04, -1.320673393369061e-04, -1.303962278828775e-04, -1.318553308867396e-04, -6.099931270043390e-04, -6.693211273820992e-04, -4.476336833810803e-08, -6.074561087687075e-08, -9.820591840318143e-04, -9.824973120195805e-04, -1.422492748935494e-03, -1.430493621012414e-03, -1.437123284490515e-03, -1.431733941149925e-03, -1.468120173796271e-03, -1.452237948776855e-03, -2.170834645782224e-03, -2.049241948475392e-03, -1.160654098394488e-03, -1.001039540273793e-03, -2.073976234447134e-03, -2.054834198513973e-03, -1.111139462866424e-03, -1.286876776773189e-03, -8.441237250633652e-03, -8.205942982733725e-03, -6.558400630138609e-03, -6.764267989349790e-03, -5.144477450235635e-03, -5.326361339215585e-03, -1.002160630749304e-05, -1.036744918149119e-05, -2.589856128399989e-03, -2.603498159193525e-03, -1.382768147455306e-07, -1.622439197302381e-07, -2.362319110942435e-03, -2.363586687356801e-03, -8.836787474508983e-03, -9.679938922949872e-03, -1.175919055941339e-02, -1.368422206231519e-02, -1.376421136307943e-03, -1.415961452968228e-03, -2.984092038850455e-05, -3.072415170679085e-05, 2.604653162702359e-16, -2.992943080225145e-15, -2.942540129190075e-06, -1.942761250264722e-06, -2.457452621388721e-03, -3.470252369076773e-03, -1.374825968126365e-13, -1.731556541154878e-12, -6.753273334096734e-06, -6.705047875815119e-06, -2.147807354195718e-12, -2.153697929420454e-12, -7.210365263393475e-15, -7.190805944536380e-15, -9.811074306774004e-05, -1.046640402260749e-04, -2.485301724124499e-11, -1.829504055543206e-11, -1.622835249208783e-06, -1.619191814398176e-06, -2.407757862937083e-14, -2.289821263066167e-14, -7.474722593275212e-03, -7.554289646277848e-03, -1.532830210793720e-03, -1.512375162004875e-03, -1.042369657172777e-03, -7.006520850340404e-04, -5.621868754259573e-03, -5.618671286072364e-03, -8.912613315002204e-04, -8.960282415676705e-04, -9.190050548611349e-04, -9.331544943740649e-04, -2.503340573638408e-07, -2.463416055910691e-07, -1.264501532321327e-07, -1.276222206931650e-07, -2.024921672271700e-05, -1.791707068752595e-05, -4.866648165216937e-03, -4.831142925216938e-03, -6.419472827971479e-03, -6.349326436938522e-03, -7.514129599516503e-03, -7.479263600387234e-03, -1.036247860766832e-06, -1.734586264389282e-06, -5.468374017869217e-03, -5.091196094744801e-03, -1.115584003336387e-02, -1.153801636881360e-02, -3.595144173790440e-07, -6.037303747260213e-07, -8.501724425596175e-03, -9.855042024265577e-03, -4.879332150946832e-11, -1.844869102516209e-10, -3.516488925543937e-11, -7.912178630117207e-12, -7.009096736887745e-13, -1.029706668635947e-12, -3.003940374896629e-03, -2.307899675448899e-04, -6.646665949495132e-12, -3.833482571479834e-11, -2.204125650082884e-12, -9.730343495012160e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_br3p86_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_br3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.993724360501327e-04, 5.993705824513621e-04, 5.993644870599172e-04, 5.993648184579209e-04, 5.993544349470468e-04, 5.993460135929415e-04, 5.994439172156385e-04, 5.994602372499094e-04, 5.993671793451756e-04, 5.994256828555093e-04, 5.993671793451756e-04, 5.994256828555093e-04, 5.790775979538670e-03, 5.792316839163780e-03, 5.790948015758742e-03, 5.792586902848386e-03, 5.796123407757472e-03, 5.797502319372218e-03, 5.786557845123779e-03, 5.788153636332443e-03, 5.790723136107049e-03, 5.792730263755496e-03, 5.790723136107049e-03, 5.792730263755496e-03, 2.182310443947541e-02, 2.234347158707085e-02, 2.165370272940338e-02, 2.228125978992071e-02, 1.954402306954552e-02, 1.888033893271545e-02, 1.929943688204852e-02, 1.954434077774750e-02, 2.309928530726783e-02, 1.622782391257831e-02, 2.309928530726783e-02, 1.622782391257831e-02, 3.752186887283547e-02, 3.886544372253475e-02, 3.844901091632360e-02, 3.997522053900644e-02, 1.813945959757951e-02, 2.055887766562883e-02, 3.137263233516051e-02, 3.159213031450663e-02, 3.725011985000711e-02, 2.807230243978503e-02, 3.725011984994575e-02, 2.807230243978501e-02, 1.779013432361062e-02, 1.790030282978575e-02, 1.815048580897864e-02, 1.831417012476843e-02, 2.179364499446075e-02, 2.214705760703494e-02, 1.600951910456450e-02, 1.584134728256936e-02, 1.725899110301585e-02, 1.939188495393593e-02, 1.725899110297825e-02, 1.939188495393604e-02, 7.632348924549730e-04, 8.767040373317271e-04, 1.445093432483482e-03, 1.453895401736840e-03, 1.068912482273501e-06, 8.446600041758177e-07, 5.433367908612535e-04, 5.282693573476245e-04, 5.215849115315102e-04, 5.274213235469586e-04, 2.439972508017356e-03, 2.677284509528397e-03, 1.790534733524321e-07, 2.429824435074830e-07, 3.928236736127255e-03, 3.929989248078322e-03, 5.689970995741975e-03, 5.721974484049652e-03, 5.748493137962061e-03, 5.726935764599699e-03, 5.872480695185083e-03, 5.808951795107421e-03, 8.683338583128896e-03, 8.196967793901567e-03, 4.642616393577951e-03, 4.004158161095173e-03, 8.295904937788534e-03, 8.219336794055891e-03, 4.444557851465697e-03, 5.147507107092757e-03, 3.376494900253458e-02, 3.282377193093490e-02, 2.623360252055443e-02, 2.705707195739916e-02, 2.057790980094254e-02, 2.130544535686232e-02, 4.008642522996854e-05, 4.146979672596477e-05, 1.035942451359996e-02, 1.041399263677410e-02, 5.531072589821222e-07, 6.489756789213080e-07, 9.449276443769732e-03, 9.454346749427205e-03, 3.534714989803592e-02, 3.871975569179949e-02, 4.703676223765354e-02, 5.473688824926073e-02, 5.505684545231772e-03, 5.663845811872911e-03, 1.193636815540182e-04, 1.228966068271634e-04, -1.041861265080944e-15, 1.197177232090058e-14, 1.177016051675684e-05, 7.771045001058886e-06, 9.829810485554882e-03, 1.388100947630709e-02, 5.499303872505462e-13, 6.926226164619512e-12, 2.701309333638694e-05, 2.682019150326047e-05, 8.591229417236454e-12, 8.614791717681816e-12, 2.884146105357390e-14, 2.876322377814552e-14, 3.924429722709600e-04, 4.186561609042995e-04, 9.941206896497995e-11, 7.318016222172822e-11, 6.491340996835133e-06, 6.476767257592705e-06, 9.631031462372281e-14, 9.159285052264670e-14, 2.989889037310085e-02, 3.021715858511141e-02, 6.131320843174876e-03, 6.049500648019499e-03, 4.169478628691106e-03, 2.802608340136162e-03, 2.248747501703830e-02, 2.247468514428946e-02, 3.565045326000880e-03, 3.584112966270682e-03, 3.676020219444540e-03, 3.732617977496258e-03, 1.001336229455363e-06, 9.853664223636940e-07, 5.058006129285306e-07, 5.104888827726599e-07, 8.099686689086800e-05, 7.166828275010381e-05, 1.946659266086775e-02, 1.932457170086775e-02, 2.567789131188592e-02, 2.539730574775409e-02, 3.005651839806601e-02, 2.991705440154893e-02, 4.144991443067329e-06, 6.938345057557130e-06, 2.187349607147687e-02, 2.036478437897920e-02, 4.462336013345549e-02, 4.615206547525431e-02, 1.438057669516176e-06, 2.414921498904085e-06, 3.400689770238469e-02, 3.942016809706231e-02, 1.951732860378733e-10, 7.379476410064834e-10, 1.406595570217575e-10, 3.164871452046883e-11, 2.803638694755098e-12, 4.118826674543789e-12, 1.201576149958652e-02, 9.231598701795595e-04, 2.658666379798053e-11, 1.533393028591934e-10, 8.816502600331535e-12, 3.892137399481234e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05