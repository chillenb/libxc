
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tca_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.214786914887393e-02, -6.214839536956513e-02, -6.214989540467969e-02, -6.214208362131853e-02, -6.214629619743051e-02, -6.214629619743051e-02, -5.209078244820875e-02, -5.209473620701392e-02, -5.218630801696354e-02, -5.202799771147823e-02, -5.209561981180017e-02, -5.209561981180017e-02, -3.071718069299626e-02, -3.050455896965470e-02, -2.534395281756012e-02, -2.563519032858759e-02, -2.567057106671438e-02, -2.567057106671438e-02, -5.447946999993982e-03, -5.864580058331250e-03, -3.410394009832072e-02, -2.083585060875445e-03, -2.617048486087661e-03, -2.617048486087660e-03, -1.275764649012195e-07, -1.556684125997387e-07, -2.973615981789617e-05, -1.991522462227943e-08, -4.007630578412256e-08, -4.007630578412261e-08, -6.438891534210743e-02, -6.449949616422250e-02, -6.439348706665358e-02, -6.449111541932510e-02, -6.444524542389894e-02, -6.444524542389894e-02, -3.100511056823426e-02, -3.151079373311589e-02, -2.991663417203504e-02, -3.035537712848875e-02, -3.182569974733451e-02, -3.182569974733451e-02, -3.913048344515019e-02, -5.175633070695887e-02, -3.632140952956320e-02, -4.790149724825187e-02, -4.068187672639021e-02, -4.068187672639024e-02, -5.872894095033086e-04, -3.395337553024758e-03, -4.763511641272188e-04, -6.658652718990010e-02, -1.198432941429833e-03, -1.198432941429833e-03, -9.931738730837279e-09, -2.030074035065174e-08, -2.409674057535377e-08, -1.584604560855835e-04, -2.678036735898042e-08, -2.678036735898041e-08, -5.351184729929459e-02, -5.054262823546964e-02, -5.160169721273428e-02, -5.247094060034602e-02, -5.203777439062160e-02, -5.203777439062160e-02, -5.378137710317406e-02, -2.799805930382998e-02, -3.461802220880891e-02, -4.224400880485338e-02, -3.832437803274089e-02, -3.832437803274089e-02, -5.206891474282311e-02, -5.840222702043242e-03, -9.713553636757246e-03, -2.210653127386409e-02, -1.499668185369776e-02, -1.499668185369777e-02, -2.670832068432916e-02, -2.445512814211546e-05, -6.732995117867217e-05, -2.591586794892022e-02, -3.848677462372692e-04, -3.848677462372691e-04, -3.496134433920473e-07, -6.564134725793366e-10, -4.966951537191073e-09, -3.124960190011859e-04, -2.403275377538444e-08, -2.403275377538440e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tca_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.094993500783124e-01, -1.094995101352783e-01, -1.094990119255840e-01, -1.094992818256476e-01, -1.094984980104465e-01, -1.094981996084113e-01, -1.095020786109260e-01, -1.095030253165678e-01, -1.094994925179280e-01, -1.095010766604489e-01, -1.094994925179280e-01, -1.095010766604489e-01, -1.050211421557008e-01, -1.050225636255913e-01, -1.050208133589304e-01, -1.050224366517597e-01, -1.050172582681906e-01, -1.050151095404367e-01, -1.050273213074499e-01, -1.050258609543225e-01, -1.050412433514618e-01, -1.050022656125412e-01, -1.050412433514618e-01, -1.050022656125412e-01, -7.776628075089720e-02, -7.752324397873690e-02, -7.758786342698946e-02, -7.729142851244573e-02, -7.092078415639654e-02, -7.127118855139722e-02, -7.164343290246111e-02, -7.153248284090177e-02, -6.998718401950452e-02, -7.335691473335978e-02, -6.998718401950452e-02, -7.335691473335978e-02, -2.050299798291175e-02, -2.008679419429487e-02, -2.188117335985732e-02, -2.137137596925025e-02, -8.359492144323651e-02, -8.133803650228942e-02, -8.456591673105187e-03, -8.399468751662214e-03, -9.721176460473624e-03, -1.377494970425994e-02, -9.721176460473619e-03, -1.377494970425993e-02, -5.689827742411838e-07, -5.534477526631016e-07, -6.957297704786531e-07, -6.742333827975796e-07, -1.312841597257112e-04, -1.280363808067459e-04, -8.722714185806920e-08, -8.789655188985635e-08, -1.676991092509166e-07, -2.219982437050185e-07, -1.676991092509169e-07, -2.219982437050188e-07, -1.021449364585279e-01, -1.021757874002060e-01, -1.020446175368854e-01, -1.020763367528458e-01, -1.021405495348153e-01, -1.021719768571681e-01, -1.020525915587831e-01, -1.020835930448485e-01, -1.020938692544081e-01, -1.021253131067637e-01, -1.020938692544081e-01, -1.021253131067637e-01, -8.672153161918664e-02, -8.672466515885699e-02, -8.748551453244421e-02, -8.750266762634573e-02, -8.509019887349359e-02, -8.498680096521093e-02, -8.580160934694264e-02, -8.569231787232051e-02, -8.780130903112210e-02, -8.808067732855383e-02, -8.780130903112210e-02, -8.808067732855383e-02, -7.785448724086565e-02, -7.806195875102569e-02, -7.037136407686591e-02, -7.032438204962412e-02, -7.753288350101174e-02, -7.490790389965191e-02, -7.183765872249678e-02, -6.870396144866078e-02, -7.587254367880238e-02, -8.003914114914987e-02, -7.587254367880242e-02, -8.003914114914989e-02, -2.489965028325795e-03, -2.478816493354634e-03, -1.337430469652207e-02, -1.334049909900927e-02, -2.050240050677065e-03, -2.000287730644362e-03, -8.527556054241960e-02, -8.533540429613136e-02, -5.040796723509829e-03, -4.902330209896849e-03, -5.040796723509829e-03, -4.902330209896849e-03, -4.407609395881078e-08, -4.331075654469759e-08, -8.956170162558975e-08, -8.895170876146818e-08, -1.074616136044728e-07, -1.046912530763407e-07, -6.841742698664423e-04, -6.822315368310679e-04, -1.308101386392209e-07, -1.126982186322818e-07, -1.308101386392208e-07, -1.126982186322817e-07, -6.399641445618207e-02, -6.358348987173183e-02, -6.856968199179256e-02, -6.817574673307811e-02, -6.704377555691757e-02, -6.664172721312803e-02, -6.570194463862444e-02, -6.529685558768197e-02, -6.638032117167474e-02, -6.597676642334613e-02, -6.638032117167474e-02, -6.597676642334613e-02, -6.203684174103551e-02, -6.168185520253687e-02, -7.081331635597092e-02, -7.057124445072140e-02, -7.500474775804830e-02, -7.469815578871555e-02, -7.413772637482421e-02, -7.384115927190830e-02, -7.530651050740533e-02, -7.502230771457723e-02, -7.530651050740533e-02, -7.502230771457723e-02, -7.202252641849892e-02, -7.185296957162873e-02, -2.190902631516348e-02, -2.184158886966785e-02, -3.382485676954469e-02, -3.363097142454958e-02, -5.963568518311919e-02, -5.939188899513677e-02, -4.682790659542438e-02, -4.683848910641268e-02, -4.682790659542439e-02, -4.683848910641270e-02, -6.843086364480094e-02, -6.806427967763842e-02, -1.067629388375361e-04, -1.064294118598349e-04, -2.944400457111443e-04, -2.900029242488302e-04, -6.257255472547887e-02, -6.179668506288304e-02, -1.663277981837941e-03, -1.621330069416961e-03, -1.663277981837941e-03, -1.621330069416961e-03, -1.548203369061531e-06, -1.523280976351551e-06, -2.889090276298695e-09, -2.886093133807558e-09, -2.217974932752629e-08, -2.156586097652318e-08, -1.341708803018204e-03, -1.331568889397267e-03, -1.157621617222699e-07, -1.013680289507814e-07, -1.157621617222697e-07, -1.013680289507812e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tca_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.191333175379224e-10, 2.382666350758448e-10, 1.191333175379224e-10, 1.191331385762791e-10, 2.382662771525581e-10, 1.191331385762791e-10, 1.191294614483767e-10, 2.382589228967534e-10, 1.191294614483767e-10, 1.191321820913266e-10, 2.382643641826532e-10, 1.191321820913266e-10, 1.191311668536439e-10, 2.382623337072878e-10, 1.191311668536439e-10, 1.191311668536439e-10, 2.382623337072878e-10, 1.191311668536439e-10, 8.244521057581535e-07, 1.648904211516307e-06, 8.244521057581535e-07, 8.245429645353740e-07, 1.649085929070748e-06, 8.245429645353740e-07, 8.265450509522755e-07, 1.653090101904551e-06, 8.265450509522755e-07, 8.222973347002708e-07, 1.644594669400542e-06, 8.222973347002708e-07, 8.244631994050609e-07, 1.648926398810122e-06, 8.244631994050609e-07, 8.244631994050609e-07, 1.648926398810122e-06, 8.244631994050609e-07, 1.558163237141681e-03, 3.116326474283362e-03, 1.558163237141681e-03, 1.548572303019788e-03, 3.097144606039576e-03, 1.548572303019788e-03, 1.290503950395557e-03, 2.581007900791114e-03, 1.290503950395557e-03, 1.260861838979305e-03, 2.521723677958610e-03, 1.260861838979305e-03, 1.278712144450983e-03, 2.557424288901966e-03, 1.278712144450983e-03, 1.278712144450983e-03, 2.557424288901966e-03, 1.278712144450983e-03, 1.013394483376591e-01, 2.026788966753182e-01, 1.013394483376591e-01, 1.068075395985097e-01, 2.136150791970193e-01, 1.068075395985097e-01, 8.369301577596226e-04, 1.673860315519245e-03, 8.369301577596226e-04, 7.942440120710594e-02, 1.588488024142119e-01, 7.942440120710594e-02, 7.659626890874432e-02, 1.531925378174886e-01, 7.659626890874432e-02, 7.659626890874431e-02, 1.531925378174886e-01, 7.659626890874431e-02, 5.780665322836153e-02, 1.156133064567231e-01, 5.780665322836153e-02, 6.218634050627003e-02, 1.243726810125401e-01, 6.218634050627003e-02, 5.448668203441024e-02, 1.089733640688205e-01, 5.448668203441024e-02, 4.461780896397199e-02, 8.923561792794397e-02, 4.461780896397199e-02, 5.102267950572140e-02, 1.020453590114428e-01, 5.102267950572140e-02, 5.102267950572153e-02, 1.020453590114431e-01, 5.102267950572153e-02, 1.479308612170855e-07, 2.958617224341711e-07, 1.479308612170855e-07, 1.481229901582828e-07, 2.962459803165655e-07, 1.481229901582828e-07, 1.479376837893478e-07, 2.958753675786956e-07, 1.479376837893478e-07, 1.481073473066544e-07, 2.962146946133088e-07, 1.481073473066544e-07, 1.480295544009567e-07, 2.960591088019133e-07, 1.480295544009567e-07, 1.480295544009567e-07, 2.960591088019133e-07, 1.480295544009567e-07, 6.292589459990270e-06, 1.258517891998054e-05, 6.292589459990270e-06, 6.240868015023890e-06, 1.248173603004778e-05, 6.240868015023890e-06, 6.068224265965772e-06, 1.213644853193154e-05, 6.068224265965772e-06, 6.026267857738636e-06, 1.205253571547727e-05, 6.026267857738636e-06, 6.373685024205060e-06, 1.274737004841012e-05, 6.373685024205060e-06, 6.373685024205060e-06, 1.274737004841012e-05, 6.373685024205060e-06, 4.667941503536614e-03, 9.335883007073227e-03, 4.667941503536614e-03, 4.312135897341820e-03, 8.624271794683639e-03, 4.312135897341820e-03, 6.223391356818955e-03, 1.244678271363791e-02, 6.223391356818955e-03, 7.425216092040162e-03, 1.485043218408032e-02, 7.425216092040162e-03, 4.563421594496533e-03, 9.126843188993065e-03, 4.563421594496533e-03, 4.563421594496535e-03, 9.126843188993070e-03, 4.563421594496535e-03, 5.879428489184817e-02, 1.175885697836963e-01, 5.879428489184817e-02, 4.191514788603119e-02, 8.383029577206237e-02, 4.191514788603119e-02, 6.198880093984992e-02, 1.239776018796998e-01, 6.198880093984992e-02, 2.231263428759355e-05, 4.462526857518709e-05, 2.231263428759355e-05, 9.004035189746892e-02, 1.800807037949378e-01, 9.004035189746892e-02, 9.004035189746892e-02, 1.800807037949378e-01, 9.004035189746892e-02, 5.559929390681068e-02, 1.111985878136214e-01, 5.559929390681068e-02, 5.281372834410058e-02, 1.056274566882012e-01, 5.281372834410058e-02, 3.514606531981594e-01, 7.029213063963189e-01, 3.514606531981594e-01, 7.632847675957137e-02, 1.526569535191427e-01, 7.632847675957137e-02, 1.524052125619501e-01, 3.048104251239003e-01, 1.524052125619501e-01, 1.524052125619500e-01, 3.048104251238999e-01, 1.524052125619500e-01, 5.946894084674101e-03, 1.189378816934820e-02, 5.946894084674101e-03, 6.381974600031930e-03, 1.276394920006386e-02, 6.381974600031930e-03, 6.331032388178778e-03, 1.266206477635756e-02, 6.331032388178778e-03, 6.224165135074030e-03, 1.244833027014806e-02, 6.224165135074030e-03, 6.286981836488277e-03, 1.257396367297655e-02, 6.286981836488277e-03, 6.286981836488277e-03, 1.257396367297655e-02, 6.286981836488277e-03, 6.218539581143079e-03, 1.243707916228616e-02, 6.218539581143079e-03, 6.738533064022967e-03, 1.347706612804593e-02, 6.738533064022967e-03, 7.539388508751791e-03, 1.507877701750358e-02, 7.539388508751791e-03, 8.122587967340122e-03, 1.624517593468024e-02, 8.122587967340122e-03, 7.903687466180358e-03, 1.580737493236072e-02, 7.903687466180358e-03, 7.903687466180358e-03, 1.580737493236072e-02, 7.903687466180358e-03, 3.442146213988125e-03, 6.884292427976250e-03, 3.442146213988125e-03, 3.139795466700492e-02, 6.279590933400983e-02, 3.139795466700492e-02, 2.932308985008572e-02, 5.864617970017144e-02, 2.932308985008572e-02, 2.847810773752837e-02, 5.695621547505673e-02, 2.847810773752837e-02, 3.035915662805059e-02, 6.071831325610118e-02, 3.035915662805059e-02, 3.035915662805060e-02, 6.071831325610120e-02, 3.035915662805060e-02, 9.620347605945214e-03, 1.924069521189043e-02, 9.620347605945214e-03, 4.862056582281157e-02, 9.724113164562313e-02, 4.862056582281157e-02, 5.396532617991417e-02, 1.079306523598283e-01, 5.396532617991417e-02, 4.546742699140793e-02, 9.093485398281587e-02, 4.546742699140793e-02, 1.006549870066826e-01, 2.013099740133653e-01, 1.006549870066826e-01, 1.006549870066826e-01, 2.013099740133653e-01, 1.006549870066826e-01, 5.064073133363473e-02, 1.012814626672695e-01, 5.064073133363473e-02, 1.979966292745685e-01, 3.959932585491371e-01, 1.979966292745685e-01, 1.127130811545995e-01, 2.254261623091990e-01, 1.127130811545995e-01, 9.753245184690640e-02, 1.950649036938128e-01, 9.753245184690640e-02, 1.940119026702388e-01, 3.880238053404776e-01, 1.940119026702388e-01, 1.940119026702392e-01, 3.880238053404783e-01, 1.940119026702392e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05