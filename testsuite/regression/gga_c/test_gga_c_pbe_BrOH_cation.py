
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.096222852624870e-02, -5.096307645628938e-02, -5.096548679754254e-02, -5.095290037105124e-02, -5.095968824013349e-02, -5.095968824013349e-02, -4.137929987043783e-02, -4.138446663210704e-02, -4.150415688662451e-02, -4.129652096575583e-02, -4.138551083721081e-02, -4.138551083721081e-02, -2.737721863246056e-02, -2.714254528593173e-02, -2.153389073381872e-02, -2.179187061070995e-02, -2.182887571025790e-02, -2.182887571025790e-02, -6.091349929476776e-03, -6.653938217482919e-03, -2.990896835219009e-02, -1.928493405192168e-03, -2.091256420678606e-03, -2.091256420678596e-03, -5.377928006568278e-09, -7.175662994352017e-09, -8.126067077928942e-06, -3.861638309468618e-10, -9.971813852080458e-10, -9.971813852080458e-10, -5.863865625559149e-02, -5.885013758449636e-02, -5.864736994029231e-02, -5.883406293040543e-02, -5.874624932939838e-02, -5.874624932939838e-02, -2.063708590389289e-02, -2.109794645186226e-02, -1.961041477488552e-02, -2.000434595823804e-02, -2.140910510214211e-02, -2.140910510214211e-02, -3.881409233962016e-02, -5.596330120972252e-02, -3.611394598320367e-02, -5.128980840843590e-02, -4.062797066182811e-02, -4.062797066182808e-02, -3.824957250139274e-04, -3.084453915703958e-03, -2.966203277956179e-04, -7.179566414569404e-02, -1.008716187092272e-03, -1.008716187092272e-03, -1.506757676527498e-10, -4.088721353677494e-10, -7.265977190577810e-10, -7.668697528120244e-05, -7.126905710551357e-10, -7.126905753919444e-10, -6.047292289470848e-02, -5.503148732922851e-02, -5.684484689135232e-02, -5.842513295545040e-02, -5.762563015445714e-02, -5.762563015445714e-02, -6.163822626042338e-02, -2.708885540478134e-02, -3.455757631806632e-02, -4.377056747303027e-02, -3.892664455216285e-02, -3.892664455216285e-02, -5.585811982547475e-02, -5.559581061898855e-03, -9.783078161684430e-03, -2.334132056568565e-02, -1.568610912811363e-02, -1.568610912811363e-02, -2.635439536902334e-02, -6.146946624202912e-06, -2.353290285953338e-05, -2.832069974343401e-02, -2.513151477459355e-04, -2.513151477459424e-04, -2.098412657958876e-08, -3.823687785668717e-12, -6.361598595995555e-11, -1.922282795263595e-04, -6.412756003298936e-10, -6.412755962099254e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.271555392140769e-01, -1.271556817168903e-01, -1.271565286277126e-01, -1.271567689282391e-01, -1.271597421927710e-01, -1.271594765111755e-01, -1.271437745832255e-01, -1.271446173976456e-01, -1.271518022921294e-01, -1.271532126661585e-01, -1.271518022921294e-01, -1.271532126661585e-01, -1.091254151332090e-01, -1.091268467209895e-01, -1.091314845950387e-01, -1.091331194798772e-01, -1.092762335248048e-01, -1.092740683070046e-01, -1.090288990054475e-01, -1.090274288282908e-01, -1.091533269937922e-01, -1.091140708431285e-01, -1.091533269937922e-01, -1.091140708431285e-01, -7.855060674457499e-02, -7.823500167343575e-02, -7.829385831431626e-02, -7.790857857287804e-02, -6.959627667374187e-02, -7.005982128425799e-02, -7.039162520602024e-02, -7.024523365970151e-02, -6.825888038680680e-02, -7.269379618176877e-02, -6.825888038680680e-02, -7.269379618176877e-02, -2.771437476089956e-02, -2.677721229790906e-02, -2.971847838623044e-02, -2.857014751337668e-02, -8.441925846625764e-02, -8.160645023336977e-02, -1.035318157807678e-02, -1.023247984280457e-02, -9.844729589938688e-03, -1.695393812624440e-02, -9.844729589938683e-03, -1.695393812624438e-02, -3.583856541925565e-08, -3.448675401295226e-08, -4.794560968615318e-08, -4.589098986031536e-08, -5.278677593134877e-05, -5.060523580390039e-05, -2.518914538729078e-09, -2.544489510123210e-09, -6.156395238271582e-09, -8.448808254629767e-09, -6.156395236103177e-09, -8.448808252461363e-09, -1.293638760631185e-01, -1.293965379075760e-01, -1.295464334576653e-01, -1.295800542974820e-01, -1.293711423049175e-01, -1.294044159802717e-01, -1.295329463010120e-01, -1.295658034219916e-01, -1.294567760471306e-01, -1.294900856475302e-01, -1.294567760471306e-01, -1.294900856475302e-01, -7.328259207863391e-02, -7.328573634988489e-02, -7.429720478560045e-02, -7.431443864422334e-02, -7.102750391355624e-02, -7.092432067083836e-02, -7.193877569411611e-02, -7.182957527657148e-02, -7.482948000433332e-02, -7.511071113671884e-02, -7.482948000433332e-02, -7.511071113671884e-02, -8.427643943983885e-02, -8.454122033899085e-02, -8.353555064421068e-02, -8.347961186015981e-02, -8.362151716851864e-02, -8.018998828942760e-02, -8.341921991544568e-02, -7.962254020208573e-02, -8.259971020796329e-02, -8.785553419383553e-02, -8.259971020796329e-02, -8.785553419383550e-02, -2.275034704492571e-03, -2.257180731530404e-03, -1.570472624352628e-02, -1.563738641234753e-02, -1.814805089380133e-03, -1.737963673099455e-03, -1.178741209118795e-01, -1.179448343864893e-01, -5.826101312260293e-03, -5.549796128514631e-03, -5.826101312260293e-03, -5.549796128514631e-03, -1.001253116546084e-09, -9.788319019215763e-10, -2.692978783407150e-09, -2.668814881624869e-09, -4.858621289027420e-09, -4.697831278814873e-09, -4.755658193564656e-04, -4.732163307773071e-04, -5.316996546332869e-09, -4.423137016536498e-09, -5.316996552214666e-09, -4.423137024153019e-09, -7.623973357929802e-02, -7.575416170636083e-02, -8.088696321860199e-02, -8.041628787902906e-02, -7.955153083406942e-02, -7.907396488710956e-02, -7.821399973586672e-02, -7.773507288400680e-02, -7.891081694129734e-02, -7.843259552153634e-02, -7.891081694129734e-02, -7.843259552153634e-02, -7.343675654239536e-02, -7.302060960745668e-02, -7.450370852322792e-02, -7.416420589137823e-02, -8.041905730619942e-02, -8.001099170912004e-02, -8.272143843375715e-02, -8.234834642206816e-02, -8.211434913410114e-02, -8.174655922685294e-02, -8.211434913410114e-02, -8.174655922685294e-02, -8.559597626243276e-02, -8.539416873750739e-02, -2.584093731429987e-02, -2.570962556358230e-02, -3.984583622572478e-02, -3.948199854346447e-02, -6.614809700734585e-02, -6.576281077608338e-02, -5.381704152488116e-02, -5.383559450779914e-02, -5.381704152488116e-02, -5.383559450779914e-02, -7.277094106486684e-02, -7.224373091148696e-02, -3.921586702744236e-05, -3.901016615357162e-05, -1.500158311541218e-04, -1.461472317217737e-04, -6.958715108404559e-02, -6.840800331875800e-02, -1.546863118891800e-03, -1.479198598767556e-03, -1.546863118891831e-03, -1.479198598767570e-03, -1.382157421732855e-07, -1.350164472635615e-07, -2.526619021841708e-11, -2.523447860591449e-11, -4.266609206703573e-10, -4.118374596353167e-10, -1.172573335489260e-03, -1.157096128244261e-03, -4.708291792616839e-09, -3.992781456307079e-09, -4.708291790163832e-09, -3.992781453637231e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.927783789727922e-10, 3.855567579455844e-10, 1.927783789727922e-10, 1.927818781952896e-10, 3.855637563905792e-10, 1.927818781952896e-10, 1.927867045996678e-10, 3.855734091993355e-10, 1.927867045996678e-10, 1.927348747403949e-10, 3.854697494807899e-10, 1.927348747403949e-10, 1.927635542656713e-10, 3.855271085313426e-10, 1.927635542656713e-10, 1.927635542656713e-10, 3.855271085313426e-10, 1.927635542656713e-10, 1.079439578828214e-06, 2.158879157656427e-06, 1.079439578828214e-06, 1.079661763830062e-06, 2.159323527660124e-06, 1.079661763830062e-06, 1.084685834641615e-06, 2.169371669283230e-06, 1.084685834641615e-06, 1.074971627900901e-06, 2.149943255801801e-06, 1.074971627900901e-06, 1.079578284035661e-06, 2.159156568071321e-06, 1.079578284035661e-06, 1.079578284035661e-06, 2.159156568071321e-06, 1.079578284035661e-06, 1.793341938121058e-03, 3.586683876242116e-03, 1.793341938121058e-03, 1.779788545257344e-03, 3.559577090514688e-03, 1.779788545257344e-03, 1.438022690962080e-03, 2.876045381924161e-03, 1.438022690962080e-03, 1.405484261965000e-03, 2.810968523930000e-03, 1.405484261965000e-03, 1.426346047129537e-03, 2.852692094259074e-03, 1.426346047129537e-03, 1.426346047129537e-03, 2.852692094259074e-03, 1.426346047129537e-03, 1.554451287660945e-01, 3.108902575321890e-01, 1.554451287660945e-01, 1.640413421794265e-01, 3.280826843588528e-01, 1.640413421794265e-01, 9.702070852199119e-04, 1.940414170439823e-03, 9.702070852199119e-04, 1.129474830562118e-01, 2.258949661124237e-01, 1.129474830562118e-01, 9.411232208644353e-02, 1.882246441728870e-01, 9.411232208644353e-02, 9.411232208644361e-02, 1.882246441728873e-01, 9.411232208644361e-02, 4.235666361990832e-03, 8.471332723898200e-03, 4.235666361990832e-03, 4.982257697020214e-03, 9.964515393423459e-03, 4.982257697020214e-03, 2.563470503076733e-02, 5.126941006152715e-02, 2.563470503076733e-02, 1.504341724364139e-03, 3.008683449168684e-03, 1.504341724364139e-03, 2.207283134900671e-03, 4.414566270491809e-03, 2.207283134900671e-03, 2.207283135350878e-03, 4.414566270088269e-03, 2.207283135350878e-03, 2.833474004550574e-07, 5.666948009101149e-07, 2.833474004550574e-07, 2.852206611918364e-07, 5.704413223836728e-07, 2.852206611918364e-07, 2.834221788603027e-07, 5.668443577206054e-07, 2.834221788603027e-07, 2.850757130984194e-07, 5.701514261968389e-07, 2.850757130984194e-07, 2.843002305715893e-07, 5.686004611431787e-07, 2.843002305715893e-07, 2.843002305715893e-07, 5.686004611431787e-07, 2.843002305715893e-07, 6.109058705422664e-06, 1.221811741084532e-05, 6.109058705422664e-06, 6.094280544391412e-06, 1.218856108878283e-05, 6.094280544391412e-06, 5.808975049122371e-06, 1.161795009824474e-05, 5.808975049122371e-06, 5.798098063791332e-06, 1.159619612758267e-05, 5.798098063791332e-06, 6.251159177461973e-06, 1.250231835492394e-05, 6.251159177461973e-06, 6.251159177461973e-06, 1.250231835492394e-05, 6.251159177461973e-06, 5.902261988533978e-03, 1.180452397706796e-02, 5.902261988533978e-03, 7.449691432044010e-03, 1.489938286408802e-02, 7.449691432044010e-03, 7.677889864733224e-03, 1.535577972946644e-02, 7.677889864733224e-03, 1.139429822267858e-02, 2.278859644535717e-02, 1.139429822267858e-02, 5.895812291530424e-03, 1.179162458306085e-02, 5.895812291530424e-03, 5.895812291530425e-03, 1.179162458306085e-02, 5.895812291530425e-03, 6.307443038105835e-02, 1.261488607621168e-01, 6.307443038105835e-02, 5.698475526193505e-02, 1.139695105238702e-01, 5.698475526193505e-02, 6.395562345774261e-02, 1.279112469154852e-01, 6.395562345774261e-02, 6.039435289405048e-05, 1.207887057881010e-04, 6.039435289405048e-05, 1.204736374298343e-01, 2.409472748596690e-01, 1.204736374298343e-01, 1.204736374298343e-01, 2.409472748596690e-01, 1.204736374298343e-01, 1.466773397935566e-03, 2.933546798259037e-03, 1.466773397935566e-03, 1.849572050554419e-03, 3.699144102256803e-03, 1.849572050554419e-03, 1.842554218706263e-02, 3.685108437354808e-02, 1.842554218706263e-02, 6.254112244285040e-02, 1.250822448857004e-01, 6.254112244285040e-02, 7.051803231269947e-03, 1.410360645889085e-02, 7.051803231269947e-03, 7.051803234557417e-03, 1.410360646869916e-02, 7.051803234557417e-03, 1.257173036459098e-02, 2.514346072918196e-02, 1.257173036459098e-02, 1.078532996058759e-02, 2.157065992117518e-02, 1.078532996058759e-02, 1.135228864245651e-02, 2.270457728491302e-02, 1.135228864245651e-02, 1.186932544936287e-02, 2.373865089872573e-02, 1.186932544936287e-02, 1.160509595445642e-02, 2.321019190891285e-02, 1.160509595445642e-02, 1.160509595445642e-02, 2.321019190891285e-02, 1.160509595445642e-02, 1.488634920260022e-02, 2.977269840520045e-02, 1.488634920260022e-02, 7.961386076538740e-03, 1.592277215307748e-02, 7.961386076538740e-03, 9.193749211225923e-03, 1.838749842245184e-02, 9.193749211225923e-03, 1.084865018983037e-02, 2.169730037966074e-02, 1.084865018983037e-02, 9.977488582726990e-03, 1.995497716545398e-02, 9.977488582726990e-03, 9.977488582726990e-03, 1.995497716545398e-02, 9.977488582726990e-03, 5.916323596889618e-03, 1.183264719377924e-02, 5.916323596889618e-03, 4.249315982125215e-02, 8.498631964250428e-02, 4.249315982125215e-02, 3.910986209029511e-02, 7.821972418059023e-02, 3.910986209029511e-02, 3.482041217076176e-02, 6.964082434152352e-02, 3.482041217076176e-02, 3.898175211299631e-02, 7.796350422599263e-02, 3.898175211299631e-02, 3.898175211299632e-02, 7.796350422599266e-02, 3.898175211299632e-02, 1.142975152732498e-02, 2.285950305464996e-02, 1.142975152732498e-02, 2.106514225067209e-02, 4.213028450134424e-02, 2.106514225067209e-02, 3.229033111123519e-02, 6.458066222247318e-02, 3.229033111123519e-02, 5.497485684367643e-02, 1.099497136873529e-01, 5.497485684367643e-02, 1.091005099993513e-01, 2.182010199987041e-01, 1.091005099993513e-01, 1.091005099993524e-01, 2.182010199987039e-01, 1.091005099993524e-01, 5.281295584737279e-03, 1.056259116938041e-02, 5.281295584737279e-03, 2.005765736367150e-03, 4.011531506685853e-03, 2.005765736367150e-03, 2.510381951246900e-03, 5.020763908060611e-03, 2.510381951246900e-03, 1.001344061858455e-01, 2.002688123716920e-01, 1.001344061858455e-01, 9.000943715250370e-03, 1.800188742308661e-02, 9.000943715250370e-03, 9.000943718924263e-03, 1.800188743452388e-02, 9.000943718924263e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05