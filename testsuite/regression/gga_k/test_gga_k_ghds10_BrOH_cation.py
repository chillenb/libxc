
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ghds10_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [3.028522861240060e+03, 3.028511316574342e+03, 3.028510798871105e+03, 3.028681519672294e+03, 3.028584840746636e+03, 3.028584840746636e+03, 9.166908244034067e+01, 9.166230441822968e+01, 9.151011379814479e+01, 9.180910490310487e+01, 9.166531747469485e+01, 9.166531747469485e+01, 5.171440096088782e+00, 5.188858103892259e+00, 5.691213446366300e+00, 5.734461036522720e+00, 5.788658232081700e+00, 5.788658232081700e+00, 1.024125070862286e+00, 1.012386851123701e+00, 6.453046280147472e+00, 9.841624750596549e-01, 1.051216624838896e+00, 1.051216624838896e+00, -6.372441603177763e-01, -6.318985392214520e-01, 4.168146620621911e-01, -8.699311020544024e-01, -7.955245281236739e-01, -7.955245281236751e-01, 1.662144736428827e+02, 1.659434398015527e+02, 1.662037863720697e+02, 1.659644837175985e+02, 1.660760627581983e+02, 1.660760627581983e+02, 4.602749352881433e+01, 4.604513757440517e+01, 4.700885193335454e+01, 4.702052806129667e+01, 4.557113911312855e+01, 4.557113911312855e+01, 3.291411988276073e+00, 3.026965484223829e+00, 3.011912223123188e+00, 2.592364246436864e+00, 3.289290907273958e+00, 3.289290907273959e+00, 9.208795299571060e-01, 1.408731976102744e+00, 8.556870281093140e-01, 2.049712056962349e+01, 8.352848630276212e-01, 8.352848630276212e-01, -1.094869790354476e+00, -9.406040691039430e-01, -1.496048199979712e+00, 5.545330267175036e-01, -1.256988525184595e+00, -1.256988525184595e+00, 2.528964014796597e+00, 2.648655935552158e+00, 2.606620327231659e+00, 2.571744573300005e+00, 2.589181427496383e+00, 2.589181427496383e+00, 2.387142989690589e+00, 3.032139938814221e+00, 2.833498909330760e+00, 2.639342024154143e+00, 2.732821017247955e+00, 2.732821017247955e+00, 3.289254893108924e+00, 1.671657154231324e+00, 1.776631559760022e+00, 1.832763101161869e+00, 1.786252800309650e+00, 1.786252800309650e+00, 2.676127266145636e+00, 4.417907988458324e-01, 5.745123049962572e-01, 1.549790591564030e+00, 5.873415032137324e-01, 5.873415032137325e-01, -3.866246638397031e-01, -1.971798967680398e+00, -1.470768273819537e+00, 5.631058316337004e-01, -1.340670206374572e+00, -1.340670206374573e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ghds10_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.492252373112956e+03, 2.492248588667085e+03, 2.492301398448917e+03, 2.492284184624947e+03, 2.492417366103402e+03, 2.492461801870154e+03, 2.491864750550153e+03, 2.491757114385359e+03, 2.492283676887860e+03, 2.492008731210981e+03, 2.492283676887860e+03, 2.492008731210981e+03, 5.814763698476251e+01, 5.819173691725604e+01, 5.815370706449644e+01, 5.820048949363613e+01, 5.832795649489739e+01, 5.837426860448481e+01, 5.804395383069015e+01, 5.809582154542016e+01, 5.811075444774896e+01, 5.825178281676134e+01, 5.811075444774896e+01, 5.825178281676134e+01, 2.237533758718406e+00, 2.368199013721003e+00, 2.191960397546524e+00, 2.352524492538398e+00, 1.576679450720269e+00, 1.340592609745470e+00, 1.470709500174914e+00, 1.549292499097314e+00, 2.566795639642377e+00, 1.245879906312578e-01, 2.566795639642377e+00, 1.245879906312578e-01, 2.252844503174078e-02, 7.687912830527838e-02, 5.688167041425302e-02, 1.174578141293922e-01, 2.752827584282821e+00, 3.487695965745811e+00, -3.347069165996467e-01, -3.312273635243137e-01, 2.602217349551450e-02, -5.728846568630916e-01, 2.602217349551372e-02, -5.728846568630910e-01, -1.947623520399685e+00, -1.921447742674967e+00, -1.905844146574664e+00, -1.873814874397177e+00, -1.271199270814690e+00, -1.237313280742465e+00, -2.229959887172142e+00, -2.248739531177831e+00, -2.004313974791168e+00, -2.009218813005581e+00, -2.004313974791167e+00, -2.009218813005581e+00, 1.655858036304089e+02, 1.654936816963227e+02, 1.659841144777962e+02, 1.658784726937821e+02, 1.656067818259144e+02, 1.655060799645197e+02, 1.659485817360558e+02, 1.658556647564632e+02, 1.657908116813805e+02, 1.656871789253950e+02, 1.657908116813805e+02, 1.656871789253950e+02, 3.357900511653425e+00, 3.355973765184303e+00, 4.028755560779705e+00, 4.012230417724913e+00, 1.623639975975014e+00, 2.138010289116080e+00, 2.214927250239771e+00, 2.720646793749610e+00, 4.975833956207103e+00, 3.879374123716326e+00, 4.975833956207103e+00, 3.879374123716326e+00, 2.779117088059316e+00, 2.771043840813446e+00, 3.864844757089170e+00, 3.875681182270037e+00, 2.284110251581830e+00, 2.478185937163167e+00, 3.053817330525695e+00, 3.199489045199561e+00, 3.060532047693647e+00, 2.806053567306793e+00, 3.060532047693648e+00, 2.806053567306796e+00, -7.446078764901563e-01, -7.253140381978675e-01, -3.340801365713481e-01, -3.298207482465784e-01, -7.776955761189727e-01, -7.430485604463167e-01, 2.712425409164025e+01, 2.710489929404905e+01, -5.428044676827835e-01, -3.774348739866701e-01, -5.428044676827835e-01, -3.774348739866701e-01, -2.251943694857847e+00, -2.281211678829926e+00, -2.179981264234428e+00, -2.200819742189971e+00, -1.919980427580622e+00, -1.876852450180138e+00, -8.880586332494599e-01, -9.003580288240142e-01, -2.030937227992381e+00, -1.879302326356119e+00, -2.030937227992381e+00, -1.879302326356120e+00, 3.570810580784462e+00, 3.594725792727869e+00, 3.396981358974604e+00, 3.421773014621681e+00, 3.457845998507720e+00, 3.482714059721336e+00, 3.508791464553998e+00, 3.532650548447569e+00, 3.483319930378294e+00, 3.507661196512843e+00, 3.483319930378294e+00, 3.507661196512843e+00, 3.463937606289227e+00, 3.482041870972954e+00, 1.620452531863440e+00, 1.644003151772588e+00, 2.144984765116958e+00, 2.172802714632601e+00, 2.680318383023683e+00, 2.698362765358822e+00, 2.413847596815772e+00, 2.431657651476650e+00, 2.413847596815772e+00, 2.431657651476650e+00, 4.128532171515630e+00, 4.152990477949555e+00, -1.540762791102748e-01, -1.445479718273605e-01, 1.664955838468911e-01, 1.965145704089027e-01, 1.014720734143056e+00, 1.027713238825777e+00, 5.715683281485726e-01, 5.785037252813796e-01, 5.715683281485718e-01, 5.785037252813792e-01, 1.442366709931140e+00, 1.478863888856108e+00, -1.325317198432790e+00, -1.320871578587500e+00, -1.157088468246453e+00, -1.125671754724166e+00, 1.145951159305677e+00, 1.184272533537778e+00, -7.260820802195102e-01, -6.135386977447372e-01, -7.260820802195095e-01, -6.135386977447372e-01, -1.861171327254654e+00, -1.835169867780987e+00, -2.496241200067714e+00, -2.404044600347490e+00, -2.248738403122241e+00, -2.204218774701345e+00, -7.132215974580728e-01, -7.092329388499365e-01, -1.932590657954832e+00, -1.914276637342121e+00, -1.932590657954832e+00, -1.914276637342121e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ghds10_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.293109184051877e-05, 0.000000000000000e+00, 1.293114179513549e-05, 1.293096651097146e-05, 0.000000000000000e+00, 1.293105074677587e-05, 1.293054083471756e-05, 0.000000000000000e+00, 1.293044770940055e-05, 1.293195688134498e-05, 0.000000000000000e+00, 1.293225240321300e-05, 1.293101834015489e-05, 0.000000000000000e+00, 1.293151277752772e-05, 1.293101834015489e-05, 0.000000000000000e+00, 1.293151277752772e-05, 2.996366023714674e-03, 0.000000000000000e+00, 2.996488674912871e-03, 2.996337920348051e-03, 0.000000000000000e+00, 2.996477974254268e-03, 2.995831833038986e-03, 0.000000000000000e+00, 2.995646812433001e-03, 2.995258276008431e-03, 0.000000000000000e+00, 2.995132169597408e-03, 2.997869647877723e-03, 0.000000000000000e+00, 2.994507043947534e-03, 2.997869647877723e-03, 0.000000000000000e+00, 2.994507043947534e-03, 4.200715030699055e-01, 0.000000000000000e+00, 4.151155631692661e-01, 4.220388987872136e-01, 0.000000000000000e+00, 4.159317269801374e-01, 4.495394155411678e-01, 0.000000000000000e+00, 4.589594270924217e-01, 4.445183114860597e-01, 0.000000000000000e+00, 4.416418299177148e-01, 4.057091912691514e-01, 0.000000000000000e+00, 4.937250273428979e-01, 4.057091912691514e-01, 0.000000000000000e+00, 4.937250273428979e-01, 2.572670895608398e+01, 0.000000000000000e+00, 2.294376358600035e+01, 2.479864693157502e+01, 0.000000000000000e+00, 2.177059790083632e+01, 2.771148678928619e-01, 0.000000000000000e+00, 2.509464054984751e-01, 5.659768155776936e+01, 0.000000000000000e+00, 5.431767266435860e+01, 2.259749942680681e+01, 0.000000000000000e+00, 1.463508641615167e+02, 2.259749942680681e+01, 0.000000000000000e+00, 1.463508641615166e+02, 6.324503014234192e+05, 0.000000000000000e+00, 5.271019733892687e+05, 5.496562727421005e+05, 0.000000000000000e+00, 4.471116735888455e+05, 3.197159921025622e+03, 0.000000000000000e+00, 2.714904401504723e+03, 2.911560192811907e+06, 0.000000000000000e+00, 3.062094217456303e+06, 8.908246190359765e+05, 0.000000000000000e+00, 4.834051151218385e+06, 8.908246190359765e+05, 0.000000000000000e+00, 4.834051151218379e+06, 8.924768742371638e-04, 0.000000000000000e+00, 8.931185292667896e-04, 8.919770600555195e-04, 0.000000000000000e+00, 8.926352803062284e-04, 8.924464270475136e-04, 0.000000000000000e+00, 8.931000056290367e-04, 8.920175841013226e-04, 0.000000000000000e+00, 8.926610178974752e-04, 8.922228825623769e-04, 0.000000000000000e+00, 8.928761159884677e-04, 8.922228825623769e-04, 0.000000000000000e+00, 8.928761159884677e-04, 1.681014771245246e-02, 0.000000000000000e+00, 1.681269628863544e-02, 1.644412702125274e-02, 0.000000000000000e+00, 1.645755969611644e-02, 1.712573498435997e-02, 0.000000000000000e+00, 1.703717992720586e-02, 1.680069107032818e-02, 0.000000000000000e+00, 1.671020204194981e-02, 1.637657215386190e-02, 0.000000000000000e+00, 1.659362732978037e-02, 1.637657215386190e-02, 0.000000000000000e+00, 1.659362732978037e-02, 6.214155513929877e-01, 0.000000000000000e+00, 6.263773909722783e-01, 4.869557603842461e-01, 0.000000000000000e+00, 4.862931574736049e-01, 8.325516992461210e-01, 0.000000000000000e+00, 7.470881093773013e-01, 7.371961063213970e-01, 0.000000000000000e+00, 6.683382188279615e-01, 5.523843371812366e-01, 0.000000000000000e+00, 6.439399747262756e-01, 5.523843371812368e-01, 0.000000000000000e+00, 6.439399747262758e-01, 1.680322502110340e+02, 0.000000000000000e+00, 1.633153648043942e+02, 2.432715455530025e+01, 0.000000000000000e+00, 2.396654094362242e+01, 2.269837421716149e+02, 0.000000000000000e+00, 1.940019561885622e+02, 1.780193950795826e-02, 0.000000000000000e+00, 1.782595457225113e-02, 1.121634537776805e+02, 0.000000000000000e+00, 9.435182145012413e+01, 1.121634537776805e+02, 0.000000000000000e+00, 9.435182145012413e+01, 6.896246147170695e+06, 0.000000000000000e+00, 6.144157416218541e+06, 3.268601510341825e+06, 0.000000000000000e+00, 3.124571891844912e+06, 7.817219853849012e+06, 0.000000000000000e+00, 6.581441351138420e+06, 6.939684423521389e+02, 0.000000000000000e+00, 6.813232572463984e+02, 7.337213909483019e+06, 0.000000000000000e+00, 2.832798288875730e+06, 7.337213909483017e+06, 0.000000000000000e+00, 2.832798288875732e+06, 6.184655642516127e-01, 0.000000000000000e+00, 6.113483131048588e-01, 6.292686936810393e-01, 0.000000000000000e+00, 6.219547129496448e-01, 6.254503941826527e-01, 0.000000000000000e+00, 6.181833347862691e-01, 6.222962328002316e-01, 0.000000000000000e+00, 6.151314409845747e-01, 6.238720007562847e-01, 0.000000000000000e+00, 6.166568959141232e-01, 6.238720007562847e-01, 0.000000000000000e+00, 6.166568959141232e-01, 6.710491118734169e-01, 0.000000000000000e+00, 6.644380041926748e-01, 1.062460454592638e+00, 0.000000000000000e+00, 1.048770501185330e+00, 9.254286161076486e-01, 0.000000000000000e+00, 9.132160890872123e-01, 8.105979508028869e-01, 0.000000000000000e+00, 8.021068030259340e-01, 8.663681416474144e-01, 0.000000000000000e+00, 8.567845172274138e-01, 8.663681416474144e-01, 0.000000000000000e+00, 8.567845172274138e-01, 4.233638525448552e-01, 0.000000000000000e+00, 4.213009399359828e-01, 1.207679180856774e+01, 0.000000000000000e+00, 1.186942020676453e+01, 7.016998509778441e+00, 0.000000000000000e+00, 6.810034028331159e+00, 3.024385837125076e+00, 0.000000000000000e+00, 2.974767882753505e+00, 4.558073793085399e+00, 0.000000000000000e+00, 4.562901000858759e+00, 4.558073793085403e+00, 0.000000000000000e+00, 4.562901000858760e+00, 1.366818081411902e+00, 0.000000000000000e+00, 1.338966339251527e+00, 3.383498365853440e+03, 0.000000000000000e+00, 3.314985802507890e+03, 1.390172763857115e+03, 0.000000000000000e+00, 1.259415447184266e+03, 3.424824182712280e+00, 0.000000000000000e+00, 3.274449679751296e+00, 3.625707363724646e+02, 0.000000000000000e+00, 3.079851680605962e+02, 3.625707363724646e+02, 0.000000000000000e+00, 3.079851680605963e+02, 2.160802635499018e+05, 0.000000000000000e+00, 1.941851121390783e+05, 1.678524443444068e+08, 0.000000000000000e+00, 1.667067706068541e+08, 1.992427887587956e+07, 0.000000000000000e+00, 1.656041588063657e+07, 4.122466915137551e+02, 0.000000000000000e+00, 3.926655637176916e+02, 8.633543549065115e+06, 0.000000000000000e+00, 3.678861720624852e+06, 8.633543549065135e+06, 0.000000000000000e+00, 3.678861720624864e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05